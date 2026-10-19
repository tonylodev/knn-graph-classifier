import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import os
from collections import Counter, deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'


def load_graph(filepath):
    G = nx.Graph()
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G


def extract_features(G):
    nodes = sorted(G.nodes())
    deg = dict(G.degree())
    cc = nx.clustering(G)
    avg_nd = nx.average_neighbor_degree(G)

    X = []
    for n in nodes:
        X.append([cc[n], avg_nd[n]])

    # label: high degree (above median) = 1, low degree = 0
    degrees = [deg[n] for n in nodes]
    median_deg = np.median(degrees)
    y = [1 if deg[n] > median_deg else 0 for n in nodes]

    return np.array(X), np.array(y)


def extract_features_fast(G):
    """Extract a low-cost feature without reusing degree as an input.

    Degree defines the synthetic high/low-degree target, so including degree in
    the feature matrix would leak the answer into the scalability benchmark.
    Average neighbor degree remains inexpensive to compute and independent of
    the target definition for the node being classified.
    """
    nodes = sorted(G.nodes())
    deg = dict(G.degree())
    avg_nd = nx.average_neighbor_degree(G)

    X = []
    for n in nodes:
        X.append([avg_nd[n]])

    degrees = [deg[n] for n in nodes]
    median_deg = np.median(degrees)
    y = [1 if deg[n] > median_deg else 0 for n in nodes]

    return np.array(X), np.array(y)


def bfs_sample(G, start_node, size):
    """Return up to ``size`` nodes discovered by breadth-first search."""
    visited = {start_node}
    queue = deque([start_node])

    while len(visited) < size and queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= size:
                    break

    return visited


def normalize(X_train, X_test):
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1
    return (X_train - mins) / rng, (X_test - mins) / rng


def knn_predict(X_train, y_train, X_test, k):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if k > len(X_train):
        raise ValueError("k cannot exceed the number of training samples")

    preds = []
    for i in range(len(X_test)):
        dists = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))
        idx = np.argsort(dists)[:k]
        neighbors = y_train[idx]
        counts = Counter(neighbors.tolist())
        preds.append(counts.most_common(1)[0][0])
    return np.array(preds)


def compute_f1(y_true, y_pred):
    """Return Macro-F1 and Micro-F1 for aligned single-label predictions.

    Include classes present in either input, assigning zero F1 when a class
    has no true positives. Reject incomplete predictions before zip can
    silently truncate them and produce misleading scores.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays")
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must contain the same non-zero number of samples")

    classes = sorted(set(y_true) | set(y_pred))

    f1_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = sum(f1_scores) / len(f1_scores)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0

    return macro_f1, micro_f1


def split_data(X, y, test_ratio=0.3, seed=42):
    """Create a reproducible stratified holdout without changing global RNG state."""
    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) != len(y) or len(y) == 0:
        raise ValueError("X and y must contain the same non-zero number of samples")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for class_label in np.unique(y):
        class_indices = np.flatnonzero(y == class_label)
        if len(class_indices) < 2:
            raise ValueError("each class needs at least two samples for a holdout split")

        rng.shuffle(class_indices)
        test_size = int(round(len(class_indices) * test_ratio))
        test_size = min(max(test_size, 1), len(class_indices) - 1)
        test_parts.append(class_indices[:test_size])
        train_parts.append(class_indices[test_size:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def stratified_k_fold_indices(y, n_splits=5, seed=42):
    """Return deterministic folds that preserve each class distribution."""
    y = np.asarray(y)

    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    if not isinstance(n_splits, (int, np.integer)) or isinstance(n_splits, bool):
        raise ValueError("n_splits must be an integer")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    classes, class_counts = np.unique(y, return_counts=True)
    if np.any(class_counts < n_splits):
        raise ValueError("each class needs at least n_splits samples")

    rng = np.random.default_rng(seed)
    test_folds = [[] for _ in range(n_splits)]

    for class_label in classes:
        class_indices = np.flatnonzero(y == class_label)
        rng.shuffle(class_indices)
        for fold_index, chunk in enumerate(np.array_split(class_indices, n_splits)):
            test_folds[fold_index].extend(chunk.tolist())

    all_indices = np.arange(len(y))
    folds = []
    for test_fold in test_folds:
        test_idx = np.asarray(test_fold, dtype=int)
        rng.shuffle(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
        folds.append((train_idx, test_idx))

    return folds


def evaluate_cross_validation(X, y, k_values, n_splits=5, seed=42):
    """Evaluate each k with fold-specific normalization and return summaries."""
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")

    folds = stratified_k_fold_indices(y, n_splits=n_splits, seed=seed)
    smallest_training_fold = min(len(train_idx) for train_idx, _ in folds)
    results = []

    for k in k_values:
        if k > smallest_training_fold:
            raise ValueError("k cannot exceed the number of samples in a training fold")

        macro_folds = []
        micro_folds = []
        time_folds = []

        for train_idx, test_idx in folds:
            X_train_n, X_test_n = normalize(X[train_idx], X[test_idx])

            t0 = time.perf_counter()
            y_pred = knn_predict(X_train_n, y[train_idx], X_test_n, k)
            elapsed = time.perf_counter() - t0

            macro_f1, micro_f1 = compute_f1(
                y[test_idx].tolist(),
                y_pred.tolist(),
            )
            macro_folds.append(macro_f1)
            micro_folds.append(micro_f1)
            time_folds.append(elapsed)

        results.append({
            'k': k,
            'macro_mean': float(np.mean(macro_folds)),
            'macro_std': float(np.std(macro_folds)),
            'micro_mean': float(np.mean(micro_folds)),
            'micro_std': float(np.std(micro_folds)),
            'time_mean': float(np.mean(time_folds)),
            'time_std': float(np.std(time_folds)),
        })

    return results


def evaluate_majority_baseline(y, n_splits=5, seed=42):
    """Evaluate a train-fold majority classifier on the same stratified folds."""
    y = np.asarray(y)
    folds = stratified_k_fold_indices(y, n_splits=n_splits, seed=seed)
    macro_folds = []
    micro_folds = []

    for train_idx, test_idx in folds:
        classes, counts = np.unique(y[train_idx], return_counts=True)
        majority_class = classes[np.argmax(counts)]
        predictions = np.full(len(test_idx), majority_class)
        macro_f1, micro_f1 = compute_f1(
            y[test_idx].tolist(),
            predictions.tolist(),
        )
        macro_folds.append(macro_f1)
        micro_folds.append(micro_f1)

    return {
        'macro_mean': float(np.mean(macro_folds)),
        'macro_std': float(np.std(macro_folds)),
        'micro_mean': float(np.mean(micro_folds)),
        'micro_std': float(np.std(micro_folds)),
    }


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Small Dataset: CA-GrQc
    print("=" * 50)
    print("Loading CA-GrQc dataset...")
    G_small = load_graph(DATA_DIR / 'CA-GrQc.txt')
    print("Nodes:", G_small.number_of_nodes(), " Edges:", G_small.number_of_edges())

    print("Extracting features...")
    X, y = extract_features(G_small)
    print("Feature matrix shape:", X.shape)
    print("Class distribution:", Counter(y.tolist()))

    # effectiveness test
    print("\n--- Effectiveness Test (CA-GrQc, 5-fold stratified CV) ---")
    k_values = [1, 3, 5, 7, 9]
    cv_results = evaluate_cross_validation(X, y, k_values, n_splits=5, seed=42)
    baseline = evaluate_majority_baseline(y, n_splits=5, seed=42)
    macro_scores = [result['macro_mean'] for result in cv_results]
    macro_stds = [result['macro_std'] for result in cv_results]
    micro_scores = [result['micro_mean'] for result in cv_results]
    micro_stds = [result['micro_std'] for result in cv_results]
    run_times = [result['time_mean'] for result in cv_results]
    run_time_stds = [result['time_std'] for result in cv_results]

    for result in cv_results:
        print(
            "k=%d: Macro-F1=%.4f +/- %.4f, Micro-F1=%.4f +/- %.4f, "
            "Mean inference time=%.2fs"
            % (
                result['k'],
                result['macro_mean'],
                result['macro_std'],
                result['micro_mean'],
                result['micro_std'],
                result['time_mean'],
            )
        )
    print(
        "Majority baseline: Macro-F1=%.4f +/- %.4f, "
        "Micro-F1=%.4f +/- %.4f"
        % (
            baseline['macro_mean'],
            baseline['macro_std'],
            baseline['micro_mean'],
            baseline['micro_std'],
        )
    )

    # plot effectiveness
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        k_values,
        macro_scores,
        yerr=macro_stds,
        fmt='o-',
        capsize=4,
        label='Macro-F1',
    )
    ax.errorbar(
        k_values,
        micro_scores,
        yerr=micro_stds,
        fmt='s-',
        capsize=4,
        label='Micro-F1',
    )
    ax.axhline(
        baseline['macro_mean'],
        color='tab:blue',
        linestyle='--',
        alpha=0.7,
        label='Majority baseline (Macro-F1)',
    )
    ax.axhline(
        baseline['micro_mean'],
        color='tab:orange',
        linestyle='--',
        alpha=0.7,
        label='Majority baseline (Micro-F1)',
    )
    ax.set_xticks(k_values)
    ax.set_xlabel('k')
    ax.set_ylabel('F1 Score')
    ax.set_title('k-NN Effectiveness on CA-GrQc (5-fold CV)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'effectiveness.png', dpi=150)
    plt.close()
    print("Saved results/effectiveness.png")

    # plot efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [str(k) for k in k_values],
        run_times,
        yerr=run_time_stds,
        capsize=4,
        color='steelblue',
    )
    ax.set_xlabel('k')
    ax.set_ylabel('Mean Inference Time per Fold (seconds)')
    ax.set_title('k-NN Efficiency on CA-GrQc (5-fold CV)')
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'efficiency.png', dpi=150)
    plt.close()
    print("Saved results/efficiency.png")

    # Large Dataset: com-dblp (Scalability) 
    print("\n" + "=" * 50)
    print("Loading com-dblp dataset for scalability test...")
    large_dataset = DATA_DIR / 'com-dblp.ungraph.txt'
    if not large_dataset.exists():
        print("Optional dataset not found; skipping scalability benchmark.")
        print("To enable it, place com-dblp.ungraph.txt in the data/ directory.")
        print("\nDone. Check the results/ folder for plots.")
        raise SystemExit(0)

    G_large = load_graph(large_dataset)
    print("Nodes:", G_large.number_of_nodes(), " Edges:", G_large.number_of_edges())

    sample_sizes = [2000, 5000, 10000]
    scale_times = []
    scale_macro = []
    scale_micro = []
    actual_sizes = []

    # pick a high-degree node as starting point for BFS sampling
    start_node = max(G_large.nodes(), key=lambda n: G_large.degree(n))

    for size in sample_sizes:
        print("\nBFS sampling %d nodes..." % size)
        visited = bfs_sample(G_large, start_node, size)

        G_sub = G_large.subgraph(list(visited)).copy()
        # remove isolated nodes just in case
        isolated = [n for n in G_sub.nodes() if G_sub.degree(n) == 0]
        G_sub.remove_nodes_from(isolated)
        print("  Nodes: %d, Edges: %d" % (G_sub.number_of_nodes(), G_sub.number_of_edges()))

        if G_sub.number_of_nodes() < 50:
            print("  Too few nodes, skipping...")
            continue

        t_start = time.time()
        X_s, y_s = extract_features_fast(G_sub)
        X_tr, X_te, y_tr, y_te = split_data(X_s, y_s)
        X_tr_n, X_te_n = normalize(X_tr, X_te)

        y_pred_s = knn_predict(X_tr_n, y_tr, X_te_n, 5)
        t_total = time.time() - t_start

        macro_f1, micro_f1 = compute_f1(y_te.tolist(), y_pred_s.tolist())
        scale_times.append(t_total)
        scale_macro.append(macro_f1)
        scale_micro.append(micro_f1)
        actual_sizes.append(G_sub.number_of_nodes())

        print("  k=5: Macro-F1=%.4f, Micro-F1=%.4f, Time=%.2fs" % (macro_f1, micro_f1, t_total))

    # plot scalability
    if scale_times:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(actual_sizes, scale_times, 'o-', color='steelblue')
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Running Time (seconds)')
        ax1.set_title('Scalability: Running Time')
        ax1.grid(True)

        ax2.plot(actual_sizes, scale_macro, 'o-', label='Macro-F1')
        ax2.plot(actual_sizes, scale_micro, 's-', label='Micro-F1')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Scalability: F1 Scores')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'scalability.png', dpi=150)
        plt.close()
        print("Saved results/scalability.png")

    print("\nAll done. Check the results/ folder for plots.")
