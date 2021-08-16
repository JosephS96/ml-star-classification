from classifiers.BaseClassifier import BaseClassifier


class AdaBoostClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()

    def fit(self, x, y, epochs, batch_size):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y, print_metrics=True):
        pass