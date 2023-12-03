from classifiers.BaseClassifier import BaseClassifier
from util.distances import *

import numpy as np


class KnnClassifier():
    def __init__(self, n_neighbors=5, distance='euclidean'):
        self.distance_metric = distance
        self.k = n_neighbors
        self.data = None
        self.labels = None
        self.n_labels = None

    def __calculate_distance(self, x, y):
        # Calculate distance from point A to point B
        # Assume that x and y are the same dimension array [1, 2, 3, ...]
        assert len(x) == len(y), "Arrays to compare are not the same length"

        if self.distance_metric == 'euclidean':
            return euclidean_distance(x, y)

        return 0

    def fit(self, x, y):
        self.data = np.array(x)
        self.labels = np.array(y)

        self.n_labels = len(np.unique(self.labels))

    def predict(self, x):
        # Store list of predictions (classes)
        predictions = []

        for item in x:
            # Calculate the distances from the current item to all other items in the sample data
            distances = []
            for sample, label in zip(self.data, self.labels):
                # Create array containing distance and label for each point -> [distance, label]
                distances.append([self.__calculate_distance(item, sample), label])

            # Sorting by distances
            distances = sorted(distances, key=lambda value: value[0])

            # Trimming the list to size k
            k_neighbors = distances[:self.k]

            # Counting the labels from the neighbors
            # List should be the size of the number of labels
            neighbors_count = np.zeros(shape=self.n_labels)
            for neighbor in k_neighbors:
                label_idx = neighbor[-1]
                neighbors_count[label_idx] += 1

            selected_label = np.argmax(neighbors_count)
            predictions.append(selected_label)

        return predictions

    def evaluate(self, x, y, print_metrics=True):
        prediction = self.predict(x)
        accuracy, precision, recall, f1_score = self.get_test_metrics(prediction, y, print_metrics=print_metrics)

        return accuracy, precision, recall, f1_score
