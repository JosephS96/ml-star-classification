import numpy as np

from classifiers.KnnClassifier import KnnClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report


if __name__ == "__main__":
    # Create test dataset.
    # x: (n_samples, n_features), Features
    # y: (n_samples,) Integer labels for classes
    x, y = make_classification(n_samples=100, n_classes=2)
    x_train = x[:80]
    y_train = y[:80]
    x_test = x[80:]
    y_test = y[80:]

    classifier = KnnClassifier(n_neighbors=5)

    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    print(classification_report(y_true=y_test, y_pred=y_pred))


