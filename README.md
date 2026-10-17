# Simulated Machine Learning Assessment: Iris Flower Classifier

## Task brief

Build a K-Nearest Neighbors (KNN) classification program that predicts an Iris flower species from its measurements.

### Requirements

- Load a labeled CSV dataset.
- Use sepal length, sepal width, petal length, and petal width as features.
- Standardize features before comparing distances.
- Create a stratified training and test split.
- Train a KNN classifier with configurable `k`.
- Report test accuracy and a confusion matrix.
- Accept measurements and predict Setosa, Versicolor, or Virginica.
- Validate dataset structure and user input.
- Include automated tests.

## Implementation

The project implements feature scaling and K-Nearest Neighbors directly using Python's standard library. KNN calculates Euclidean distance, selects the closest training examples, and uses majority voting to choose the species.

## Requirements

- Python 3.8 or newer

No external dependencies are required.

## Run with measurements

Supply sepal length, sepal width, petal length, and petal width in centimeters:

```bash
python3 knn_classifier.py 5.1 3.5 1.4 0.2
```

## Run interactively

```bash
python3 knn_classifier.py
```

## Change the number of neighbors

```bash
python3 knn_classifier.py 6.3 3.0 5.8 2.1 --neighbors 5
```

## Use another dataset

The CSV must contain `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, and `species` columns.

```bash
python3 knn_classifier.py 5.1 3.5 1.4 0.2 --dataset path/to/iris.csv
```

## Run automated tests

```bash
python3 -m unittest -v
```

## Educational note

The included 30-row dataset is intentionally small for assessment purposes. A production model would use a larger dataset, randomized cross-validation, and additional monitoring.
