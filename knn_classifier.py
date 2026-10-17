#!/usr/bin/env python3
"""Classify Iris flowers using K-Nearest Neighbors."""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Features = List[float]


class StandardScaler:
    """Standardize features to zero mean and unit standard deviation."""

    def __init__(self) -> None:
        self.means = []
        self.standard_deviations = []

    def fit(self, rows: Sequence[Features]) -> "StandardScaler":
        if not rows:
            raise ValueError("At least one row is required to fit the scaler.")

        feature_count = len(rows[0])
        if feature_count == 0 or any(len(row) != feature_count for row in rows):
            raise ValueError("Every row must contain the same number of features.")

        self.means = [
            sum(row[index] for row in rows) / len(rows)
            for index in range(feature_count)
        ]
        self.standard_deviations = []

        for index, mean in enumerate(self.means):
            variance = sum((row[index] - mean) ** 2 for row in rows) / len(rows)
            deviation = math.sqrt(variance)
            self.standard_deviations.append(deviation if deviation > 0 else 1.0)

        return self

    def transform_one(self, row: Features) -> Features:
        if not self.means:
            raise RuntimeError("Fit the scaler before transforming values.")
        if len(row) != len(self.means):
            raise ValueError("Feature count does not match the fitted scaler.")

        return [
            (value - mean) / deviation
            for value, mean, deviation in zip(row, self.means, self.standard_deviations)
        ]

    def transform(self, rows: Sequence[Features]) -> List[Features]:
        return [self.transform_one(row) for row in rows]


class KNearestNeighbors:
    """A simple K-Nearest Neighbors classifier using Euclidean distance."""

    def __init__(self, neighbors: int = 3) -> None:
        if neighbors < 1:
            raise ValueError("The number of neighbors must be at least 1.")
        self.neighbors = neighbors
        self.features = []
        self.labels = []

    def fit(self, features: Sequence[Features], labels: Sequence[str]) -> "KNearestNeighbors":
        if len(features) != len(labels) or not features:
            raise ValueError("Features and labels must be non-empty and equal in length.")
        if self.neighbors > len(features):
            raise ValueError("The number of neighbors cannot exceed the training rows.")

        feature_count = len(features[0])
        if feature_count == 0 or any(len(row) != feature_count for row in features):
            raise ValueError("Every training row must have the same feature count.")

        self.features = [list(row) for row in features]
        self.labels = list(labels)
        return self

    @staticmethod
    def distance(first: Features, second: Features) -> float:
        if len(first) != len(second):
            raise ValueError("Distance inputs must have the same feature count.")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))

    def predict_one(self, row: Features) -> str:
        if not self.features:
            raise RuntimeError("Train the classifier before making predictions.")
        if len(row) != len(self.features[0]):
            raise ValueError("Prediction feature count does not match training data.")

        nearest = sorted(
            (
                (self.distance(row, training_row), label)
                for training_row, label in zip(self.features, self.labels)
            ),
            key=lambda item: (item[0], item[1]),
        )[: self.neighbors]

        votes = Counter(label for _, label in nearest)
        distance_totals = defaultdict(float)
        for distance, label in nearest:
            distance_totals[label] += distance

        return min(
            votes,
            key=lambda label: (-votes[label], distance_totals[label], label),
        )

    def predict(self, rows: Sequence[Features]) -> List[str]:
        return [self.predict_one(row) for row in rows]


def load_dataset(file_path: Path) -> Tuple[List[Features], List[str]]:
    """Load four measurements and species labels from CSV."""
    required = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    if not file_path.is_file():
        raise ValueError(f"Dataset not found: {file_path}")

    features, labels = [], []
    try:
        with file_path.open(newline="", encoding="utf-8") as data_file:
            reader = csv.DictReader(data_file)
            if not reader.fieldnames or any(column not in reader.fieldnames for column in required):
                raise ValueError("Dataset is missing one or more required columns.")

            for row_number, row in enumerate(reader, start=2):
                try:
                    features.append([float(row[column]) for column in required[:4]])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid measurement on CSV row {row_number}.") from error

                label = (row["species"] or "").strip().lower()
                if not label:
                    raise ValueError(f"Missing species on CSV row {row_number}.")
                labels.append(label)
    except OSError as error:
        raise ValueError(f"Could not read dataset: {file_path}") from error

    if len(features) < 6:
        raise ValueError("Dataset must contain at least six observations.")
    return features, labels


def stratified_split(
    features: Sequence[Features],
    labels: Sequence[str],
    test_every: int = 5,
) -> Tuple[List[Features], List[Features], List[str], List[str]]:
    """Select every nth row per class for a deterministic stratified test set."""
    if len(features) != len(labels) or not features:
        raise ValueError("Features and labels must be non-empty and equal in length.")
    if test_every < 2:
        raise ValueError("test_every must be at least 2.")

    class_positions = Counter()
    x_train, x_test, y_train, y_test = [], [], [], []

    for row, label in zip(features, labels):
        class_positions[label] += 1
        if class_positions[label] % test_every == 0:
            x_test.append(list(row))
            y_test.append(label)
        else:
            x_train.append(list(row))
            y_train.append(label)

    if not x_train or not x_test:
        raise ValueError("Dataset is too small for the requested split.")
    return x_train, x_test, y_train, y_test


def accuracy(actual: Sequence[str], predicted: Sequence[str]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Actual and predicted labels must be non-empty and equal in length.")
    return sum(a == p for a, p in zip(actual, predicted)) / len(actual)


def confusion_matrix(actual: Sequence[str], predicted: Sequence[str]) -> Dict[str, Counter]:
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must have the same length.")
    matrix = defaultdict(Counter)
    for actual_label, predicted_label in zip(actual, predicted):
        matrix[actual_label][predicted_label] += 1
    return dict(matrix)


def build_parser() -> argparse.ArgumentParser:
    default_dataset = Path(__file__).resolve().parent / "data" / "iris_small.csv"
    parser = argparse.ArgumentParser(
        description="Classify an Iris flower with K-Nearest Neighbors."
    )
    parser.add_argument("measurements", nargs="*", type=float, help="Sepal length, sepal width, petal length, and petal width in cm.")
    parser.add_argument("--neighbors", type=int, default=3, help="Number of neighbors (default: 3).")
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="Path to the Iris CSV dataset.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        measurements = args.measurements
        if not measurements:
            entered = input("Enter sepal length, sepal width, petal length, and petal width: ")
            measurements = [float(value) for value in entered.replace(",", " ").split()]

        if len(measurements) != 4:
            raise ValueError("Provide exactly four flower measurements.")
        if any(not math.isfinite(value) or value <= 0 for value in measurements):
            raise ValueError("All flower measurements must be positive finite numbers.")

        features, labels = load_dataset(args.dataset)
        x_train, x_test, y_train, y_test = stratified_split(features, labels)

        evaluation_scaler = StandardScaler().fit(x_train)
        evaluation_model = KNearestNeighbors(args.neighbors).fit(
            evaluation_scaler.transform(x_train), y_train
        )
        test_predictions = evaluation_model.predict(evaluation_scaler.transform(x_test))
        model_accuracy = accuracy(y_test, test_predictions)
        matrix = confusion_matrix(y_test, test_predictions)

        final_scaler = StandardScaler().fit(features)
        final_model = KNearestNeighbors(args.neighbors).fit(
            final_scaler.transform(features), labels
        )
        prediction = final_model.predict_one(final_scaler.transform_one(measurements))
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("\nK-NEAREST NEIGHBORS REPORT")
    print("=" * 39)
    print(f"Dataset records:          {len(features)}")
    print(f"Training records:         {len(x_train)}")
    print(f"Testing records:          {len(x_test)}")
    print(f"Neighbors (k):            {args.neighbors}")
    print(f"Test accuracy:            {model_accuracy * 100:.2f}%")
    print("Confusion matrix:")
    for actual_label in sorted(matrix):
        details = ", ".join(
            f"{predicted_label}={count}"
            for predicted_label, count in sorted(matrix[actual_label].items())
        )
        print(f"  Actual {actual_label:10} -> {details}")
    print("-" * 39)
    print(f"Measurements (cm):        {', '.join(f'{value:g}' for value in measurements)}")
    print(f"Predicted species:        Iris {prediction.capitalize()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
