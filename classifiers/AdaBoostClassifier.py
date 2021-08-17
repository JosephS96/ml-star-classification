from classifiers.BaseClassifier import BaseClassifier
from sklearn.tree import DecisionTreeClassifier
import math
import numpy as np


class AdaBoostClassifier(BaseClassifier):
    def __init__(self, n_classes=6):
        super().__init__()

        self.sample_weights = []  # weights for each x entrance
        self.alphas = []  # weights for the classifiers
        self.classifiers = []  # the trained classifiers

        self.n_classes = n_classes

    """Error rate of the candidate is defined as the sum of the weights from teh misclassified sample"""
    def calculate_error_rate(self, x, y):
        error_sum = 0
        # Check the predictions against the ground truth
        for i in range(x):
            if x[i] != y[i]:
                error_sum += self.sample_weights[i]

        return error_sum / np.sum(self.sample_weights)

    """Calculate weight for the given classifier
        This is the main change for multi-class adaboosting taken from the Stanford paper "Multi-class AdaBoost"
    """
    def calculate_alpha(self, error):
        return math.log((1 - error) / error) + math.log(self.n_classes - 1)

    def update_weights(self, prediction, ground_truth, error):
        for i in range(len(prediction)):
            if prediction[i] == ground_truth[i]:
                self.sample_weights[i] = 0.5 * (self.sample_weights[i] / (1 - error))
            else:
                self.sample_weights[i] = 0.5 * (self.sample_weights[i] / error)

            # self.sample_weights[i] = self.sample_weights[i] * np.exp(-alpha * ground_truth[i] * prediction[i])

    def fit(self, x, y, epochs, batch_size):
        # Set initial weights
        initial_weight = 1 / len(x)
        self.sample_weights = [initial_weight] * len(x)

        for epoch in range(epochs):
            # Fit current learner
            weak_classifier = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            weak_classifier.fit(x, y, sample_weight=self.sample_weights)

            # Calculate error and stump weight from weak learner prediction
            weak_prediction = weak_classifier.predict(x)
            candidate_error = self.calculate_error_rate(weak_prediction, y)
            alpha = self.calculate_alpha(candidate_error)

            # Store the classifier and the alpha value
            self.classifiers.append(weak_classifier)
            self.alphas.append(alpha)

            # updating weights for the next iteration (next classifier)
            self.update_weights(weak_prediction, y, candidate_error)

            # Save metrics for history visualization
            self.history.save_metrics(self.predict(x), y)

    def predict(self, x):
        # Store list of predictions (classes)
        predictions = []

        for item in x:
            classes = [0] * self.n_classes
            for k in range(self.n_classes):
                for m in range(len(self.classifiers)):
                    # Predict receives an array
                    prediction = self.classifiers[m].predict([item])
                    # If the classifiers outputs the desired class, sum the alphas
                    if prediction == k:
                        classes[k] += self.alphas[m]

            # Once that I have tried all the classes with the classifiers
            selected_class = np.argmax(classes)
            predictions.append(selected_class)

        return predictions

    def evaluate(self, x, y, print_metrics=True):
        prediction = self.predict(x)
        accuracy, precision, recall, f1_score = self.get_test_metrics(prediction, y, print_metrics=print_metrics)

        return accuracy, precision, recall, f1_score
