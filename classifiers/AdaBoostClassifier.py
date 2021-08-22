from classifiers.BaseClassifier import BaseClassifier
import math
import numpy as np

from classifiers.KnnClassifier import KnnClassifier
from classifiers.NaiveBayesClassifier import NaiveBayesClassifier


class AdaBoostClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()

        self.sample_weights = []  # weights for each x entrance
        self.alphas = []  # weights for the classifiers
        self.classifiers = []  # the trained classifiers

    """Error rate of the candidate is defined as the sum of the weights from teh misclassified sample"""
    def calculate_error_rate(self, x, y):
        error_sum = 0
        # Check the predictions against the ground truth
        for i in range(len(x)):
            if x[i] != y[i]:
                error_sum += self.sample_weights[i]

        weights_sum = np.sum(self.sample_weights)
        return error_sum / weights_sum

    """Calculate weight for the given classifier
        This is the main change for multi-class adaboosting taken from the Stanford paper "Multi-class AdaBoost"
    """
    def calculate_alpha(self, error):
        if error > 0:
            return math.log((1 - error) / error) + math.log(self.n_classes - 1)
        else:
            return 1

    def normalize_alphas(self):
        alphas_sum = 0
        for i in range(len(self.alphas)):
            alphas_sum += self.alphas[i]

        for i in range(len(self.alphas)):
            self.alphas[i] = self.alphas[i] / alphas_sum

    def update_weights(self, prediction, ground_truth, alpha):
        for i in range(len(prediction)):
            if prediction[i] != ground_truth[i]:
                self.sample_weights[i] = self.sample_weights[i] * np.exp(alpha)

        weights_sum = np.sum(self.sample_weights)

        # Re-normalize the weights so they add up to 1
        for i in range(len(self.sample_weights)):
            self.sample_weights[i] = self.sample_weights[i] / weights_sum

    def resample(self, x, y, weights):
        # Re-sample with probabilities
        n_samples = len(x)
        indexes_list = list(range(0, n_samples))
        new_sample = np.random.choice(indexes_list, n_samples, p=weights)

        # Divide into x and y again
        new_x = []
        new_y = []

        for index in new_sample:
            new_x.append(x[index])
            new_y.append(y[index])

        return new_x, new_y

    def fit(self, x, y, epochs, batch_size):
        self.set_n_classes(y)

        # Set initial weights
        initial_weight = 1 / len(x)
        self.sample_weights = [initial_weight] * len(x)

        for epoch in range(epochs):
            # Fit current learner
            # weak_classifier = DecisionTreeClassifier(max_depth=2)
            # weak_classifier.fit(x, y, sample_weight=self.sample_weights)

            new_x, new_y = self.resample(x, y, self.sample_weights)
            # weak_classifier = KnnClassifier(n_neighbors=11)
            # weak_classifier.fit(new_x, new_y, epochs=0, batch_size=0)
            weak_classifier = NaiveBayesClassifier()
            weak_classifier.fit(new_x, new_y, epochs=0, batch_size=0)

            # Evaluate weak classifier for reference
            print("Weak classifier: ")
            # weak_classifier.evaluate(new_x, new_y, print_metrics=True)

            # Calculate error and stump weight from weak learner prediction
            weak_prediction = weak_classifier.predict(x)
            candidate_error = self.calculate_error_rate(weak_prediction, y)
            alpha = self.calculate_alpha(candidate_error)

            # Store the classifier and the alpha value
            self.classifiers.append(weak_classifier)
            self.alphas.append(alpha)

            # updating weights for the next iteration (next classifier)
            self.update_weights(weak_prediction, y, alpha)

            # Normalize alphas
            # self.normalize_alphas()

            # Save metrics for history visualization
            prediction = self.predict(x)
            self.history.save_metrics(prediction, y)

        return self.get_training_history()

    def predict(self, x):
        # Store list of predictions (classes)
        classes_count = np.zeros(shape=(len(x), self.n_classes))
        for m in range(len(self.classifiers)):
            prediction = self.classifiers[m].predict(x)
            for i in range(len(prediction)):
                classes_count[i][prediction[i]] += self.alphas[m]

        # Get argmax to determine actual prediction
        predictions = np.argmax(classes_count, axis=1)

        return predictions

    def evaluate(self, x, y, print_metrics=True):
        prediction = self.predict(x)
        accuracy, precision, recall, f1_score = self.get_test_metrics(prediction, y, print_metrics=print_metrics)

        return accuracy, precision, recall, f1_score
