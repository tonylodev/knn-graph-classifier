import unittest

import networkx as nx
import numpy as np

from knn_classifier import (
    DATA_DIR,
    bfs_sample,
    evaluate_cross_validation,
    evaluate_majority_baseline,
    extract_features_fast,
    knn_predict,
    load_graph,
    split_data,
    stratified_k_fold_indices,
)


class KnnClassifierTests(unittest.TestCase):
    def test_included_dataset_metadata_matches_loaded_graph(self):
        graph = load_graph(DATA_DIR / 'CA-GrQc.txt')

        self.assertEqual(graph.number_of_nodes(), 5242)
        self.assertEqual(graph.number_of_edges(), 14496)

    def test_scalability_features_exclude_degree_label_source(self):
        graph = nx.star_graph(4)

        features, labels = extract_features_fast(graph)

        self.assertEqual(features.shape, (5, 1))
        np.testing.assert_allclose(
            features[:, 0],
            [1.0, 4.0, 4.0, 4.0, 4.0],
        )
        np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0])

    def test_bfs_sample_respects_requested_size(self):
        graph = nx.path_graph(10)

        sampled = bfs_sample(graph, start_node=0, size=4)

        self.assertEqual(sampled, {0, 1, 2, 3})

    def test_split_data_is_reproducible_and_stratified(self):
        features = np.arange(40).reshape(20, 2)
        labels = np.array([0] * 10 + [1] * 10)

        first = split_data(features, labels, seed=7)
        second = split_data(features, labels, seed=7)

        for first_array, second_array in zip(first, second):
            np.testing.assert_array_equal(first_array, second_array)

        self.assertEqual(np.bincount(first[2]).tolist(), [7, 7])
        self.assertEqual(np.bincount(first[3]).tolist(), [3, 3])

    def test_stratified_folds_cover_every_sample_once(self):
        labels = np.array([0] * 12 + [1] * 8)

        folds = stratified_k_fold_indices(labels, n_splits=4, seed=7)

        test_indices = np.concatenate([test_idx for _, test_idx in folds])
        np.testing.assert_array_equal(np.sort(test_indices), np.arange(len(labels)))
        for train_idx, test_idx in folds:
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            self.assertEqual(np.bincount(labels[test_idx]).tolist(), [3, 2])

    def test_cross_validation_reports_one_summary_per_k(self):
        features = np.concatenate([
            np.linspace(0.0, 0.4, 10),
            np.linspace(0.6, 1.0, 10),
        ]).reshape(-1, 1)
        labels = np.array([0] * 10 + [1] * 10)

        results = evaluate_cross_validation(
            features,
            labels,
            k_values=[1, 3],
            n_splits=5,
            seed=7,
        )

        self.assertEqual([result['k'] for result in results], [1, 3])
        for result in results:
            self.assertAlmostEqual(result['macro_mean'], 1.0)
            self.assertAlmostEqual(result['micro_mean'], 1.0)
            self.assertGreaterEqual(result['time_mean'], 0.0)

    def test_majority_baseline_is_fit_on_each_training_fold(self):
        labels = np.array([0] * 12 + [1] * 8)

        result = evaluate_majority_baseline(labels, n_splits=4, seed=7)

        self.assertAlmostEqual(result['macro_mean'], 0.375)
        self.assertAlmostEqual(result['macro_std'], 0.0)
        self.assertAlmostEqual(result['micro_mean'], 0.6)
        self.assertAlmostEqual(result['micro_std'], 0.0)

    def test_knn_predicts_nearest_class(self):
        train_features = np.array([[0.0], [0.2], [0.8], [1.0]])
        train_labels = np.array([0, 0, 1, 1])
        test_features = np.array([[0.1], [0.9]])

        predictions = knn_predict(train_features, train_labels, test_features, k=3)

        np.testing.assert_array_equal(predictions, [0, 1])

    def test_knn_rejects_k_larger_than_training_set(self):
        with self.assertRaisesRegex(ValueError, 'training samples'):
            knn_predict(
                np.array([[0.0], [1.0]]),
                np.array([0, 1]),
                np.array([[0.5]]),
                k=3,
            )


if __name__ == '__main__':
    unittest.main()
