"""Tests for the K-Nearest Neighbors flower classifier."""

import tempfile
import unittest
from pathlib import Path

from knn_classifier import (
    KNearestNeighbors,
    StandardScaler,
    accuracy,
    confusion_matrix,
    load_dataset,
    stratified_split,
)


class KNNClassifierTests(unittest.TestCase):
    def test_calculates_euclidean_distance(self):
        self.assertAlmostEqual(KNearestNeighbors.distance([0, 0], [3, 4]), 5)

    def test_classifies_nearest_group(self):
        model = KNearestNeighbors(neighbors=3).fit(
            [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]],
            ["a", "a", "a", "b", "b", "b"],
        )
        self.assertEqual(model.predict_one([0.5, 0.5]), "a")
        self.assertEqual(model.predict_one([10.5, 10.5]), "b")

    def test_rejects_prediction_before_training(self):
        with self.assertRaises(RuntimeError):
            KNearestNeighbors().predict_one([1, 2])

    def test_standardizes_features(self):
        scaler = StandardScaler().fit([[1, 10], [2, 20], [3, 30]])
        transformed = scaler.transform([[1, 10], [2, 20], [3, 30]])
        self.assertAlmostEqual(sum(row[0] for row in transformed), 0)
        self.assertAlmostEqual(sum(row[1] for row in transformed), 0)

    def test_creates_stratified_split(self):
        features = [[float(value)] for value in range(1, 11)]
        labels = ["a"] * 5 + ["b"] * 5
        _, x_test, _, y_test = stratified_split(features, labels, test_every=5)
        self.assertEqual(x_test, [[5.0], [10.0]])
        self.assertEqual(y_test, ["a", "b"])

    def test_calculates_accuracy_and_confusion_matrix(self):
        actual = ["a", "a", "b"]
        predicted = ["a", "b", "b"]
        self.assertAlmostEqual(accuracy(actual, predicted), 2 / 3)
        matrix = confusion_matrix(actual, predicted)
        self.assertEqual(matrix["a"]["a"], 1)
        self.assertEqual(matrix["a"]["b"], 1)
        self.assertEqual(matrix["b"]["b"], 1)

    def test_loads_valid_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "iris.csv"
            dataset.write_text(
                "sepal_length,sepal_width,petal_length,petal_width,species\n"
                "5.1,3.5,1.4,0.2,setosa\n"
                "5.0,3.4,1.5,0.2,setosa\n"
                "6.0,3.0,4.5,1.5,versicolor\n"
                "6.1,2.9,4.7,1.4,versicolor\n"
                "6.5,3.0,5.5,2.0,virginica\n"
                "6.7,3.1,5.6,2.1,virginica\n",
                encoding="utf-8",
            )
            features, labels = load_dataset(dataset)
            self.assertEqual(len(features), 6)
            self.assertEqual(labels[-1], "virginica")

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            KNearestNeighbors(neighbors=0)
        with self.assertRaises(ValueError):
            KNearestNeighbors(neighbors=3).fit([[1], [2]], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
