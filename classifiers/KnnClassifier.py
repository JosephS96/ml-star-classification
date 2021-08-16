from classifiers.BaseClassifier import BaseClassifier
import numpy as np
import math


class KnnClassifier(BaseClassifier):
    def __init__(self, k=5):
        super().__init__()
        self.distance = 'euclidean'
        self.k = k
        self.n_classes = 6

        self.data = []
        self.labels = []

    def get_distance(self, x, y):
        # Calculate distance from point A to point B
        # Assume that x and y are the same dimension array [1, 2, 3, ...]
        dist_sum = 0
        for i in range(len(x) - 1):  # -1 excludes the target label form the count
            dist_sum += (y[i] - x[i]) ** 2

        return math.sqrt(dist_sum)

    def fit(self, x, y, epochs, batch_size):
        # Store the dataset for comparison
        for i in range(len(y)):
            x[i].append(y[i])

        self.data = x

    def predict(self, x):
        # Store list of predictions (classes)
        predictions = []

        for item in x:
            # Calculate the distances from one point to all others in the data
            distances = []

            for point in self.data:
                # Create array containing distance and label for each point -> [distance, label]
                distances.append([self.get_distance(item, point), point[-1]])

            # Sorting the distances
            distances = sorted(distances, key=lambda value: value[0])

            # Trimming the list to size k
            k_neighbors = distances[:self.k]

            # Counting the labels from the neighbors
            # List should be the size of the number of labels -> 6
            neighbors_count = []
            for i in range(self.n_classes):
                neighbors_count.append(0)

            for i in range(self.k):
                index = k_neighbors[i][-1]
                neighbors_count[index] += 1

            selected_label = np.argmax(neighbors_count)
            predictions.append(selected_label)

        return predictions

    def evaluate(self, x, y, print_metrics=True):
        prediction = self.predict(x)
        accuracy, precision, recall, f1_score = self.get_test_metrics(prediction, y, print_metrics=print_metrics)

        return accuracy, precision, recall, f1_score
