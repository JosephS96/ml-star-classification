from abc import ABC, abstractmethod

from ModelHistory import ModelHistory


class BaseClassifier(ABC):
    def __init__(self):
        self.history = ModelHistory()

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
    def evaluate(self, x, y):
        """"Evaluates the model returning all the important metrics, accuracy, f1 score and loss values
        for the given test data

        :return: ModelHistory object with the respective history of metrics for plotting
        """
        pass
