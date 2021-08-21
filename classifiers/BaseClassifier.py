from abc import ABC, abstractmethod

from ModelHistory import ModelHistory
import numpy as np


class BaseClassifier(ABC):
    def __init__(self):
        self.history = ModelHistory()
        self.n_classes = None

    @abstractmethod
    def fit(self, x, y, epochs, batch_size):
        """Adjust the data to the desired labels, at the end it should return
        a ModelHistory object with the required metrics and data to plot

        :return ModelHistory object containing the required metrics
        """
        pass

    @abstractmethod
    def predict(self, x):
        """Given a single or a batch of x values, predict the respective y value depending
        on the trained model

        :returns single or batch y predictions in array like form
        """
        pass

    @abstractmethod
    def evaluate(self, x, y, print_metrics=True):
        """"Evaluates the model returning all the important metrics, accuracy, f1 score and loss values
        for the given test data

        :return: Metrics in separate variables
        """
        pass

    def get_training_history(self):
        """Get the history of the training contained in lists for the global model (not per class)

        :returns List of values for accuracy, precision, recall and f1-score
        """
        return self.history.get_training_metrics()

    def get_test_metrics(self, predicted, ground_truth, print_metrics=True):
        return self.history.get_evaluation_report(predicted, ground_truth, print_metrics)

    def set_n_classes(self, y):
        unique_classes = set(y)
        n_classes = max(unique_classes)
        print(unique_classes)
        self.n_classes = n_classes + 1
