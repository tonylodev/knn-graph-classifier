import unittest

import numpy as np

from knn_classifier import compute_f1


class F1MetricTests(unittest.TestCase):
    def test_perfect_and_completely_wrong_predictions(self):
        for truth, prediction, expected in [
            ([0, 1, 2], [0, 1, 2], (1.0, 1.0)),
            ([0, 1, 2], [1, 2, 0], (0.0, 0.0)),
            ([1], [1], (1.0, 1.0)),
        ]:
            with self.subTest(truth=truth, prediction=prediction):
                self.assertEqual(compute_f1(truth, prediction), expected)

    def test_multiclass_scores_match_hand_calculated_confusion_matrix(self):
        # Rows are true classes, columns are predicted classes:
        # [[1, 2, 0], [0, 1, 1], [0, 0, 1]].
        # Per-class F1: 1/2, 2/5, 2/3; three of six predictions correct.
        macro, micro = compute_f1([0, 0, 0, 1, 1, 2], [0, 1, 1, 1, 2, 2])

        self.assertAlmostEqual(macro, 47 / 90)
        self.assertAlmostEqual(micro, 1 / 2)

    def test_classes_missing_from_either_side_still_count_in_macro_average(self):
        for truth, prediction in [
            ([0, 0, 1, 1], [0, 0, 0, 0]),
            ([0, 0], [0, 2]),
        ]:
            with self.subTest(truth=truth, prediction=prediction):
                macro, micro = compute_f1(truth, prediction)
                self.assertAlmostEqual(macro, 1 / 3)
                self.assertAlmostEqual(micro, 1 / 2)

    def test_micro_f1_equals_accuracy_for_single_label_classification(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        for prediction in [truth, truth[::-1], np.zeros_like(truth)]:
            with self.subTest(prediction=prediction.tolist()):
                _, micro = compute_f1(truth, prediction)
                self.assertAlmostEqual(micro, np.mean(truth == prediction))

    def test_rejects_empty_or_incomplete_predictions_instead_of_scoring_them(self):
        for truth, prediction in [
            ([], []),
            ([0], []),
            ([], [0]),
            ([0, 1], [0]),
            ([0], [0, 1]),
        ]:
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaisesRegex(ValueError, 'same non-zero number'):
                    compute_f1(truth, prediction)

    def test_rejects_scalar_and_multidimensional_labels(self):
        for truth, prediction in [
            (0, [0]),
            ([0], 0),
            ([[0], [1]], [0, 1]),
            ([0, 1], [[0], [1]]),
        ]:
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaisesRegex(ValueError, 'one-dimensional'):
                    compute_f1(truth, prediction)


if __name__ == '__main__':
    unittest.main()
