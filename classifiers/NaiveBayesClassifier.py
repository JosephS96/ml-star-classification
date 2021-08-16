from classifiers.BaseClassifier import BaseClassifier
import math
import numpy as np


# This class implements Gaussian Naive Bayes, due to the continual characteristics of the dataset
class NaiveBayesClassifier(BaseClassifier):
    def __init__(self, n_classes=6):
        super().__init__()

        self.n_classes = n_classes
        self.priors = []
        self.means = []
        self.stds = []

    """Define the prior probabilities of each class, prior probability was defined as class-count / sample size"""
    def set_priors(self, y_labels):
        labels_count = np.zeros(self.n_classes)
        for label in y_labels:
            labels_count[label] += 1

        labels_count = labels_count / len(y_labels)
        self.priors = labels_count

    """Receives features (x) and labels (y)"""
    def set_means_and_stds(self, features, labels):
        class_means = []
        class_stds = []

        # For each class in our data
        for i in range(self.n_classes):
            # I need to create an array with the sum of the classes like this -> [0, 1, 2, 3, 4, 5]
            features_sum = []
            for j in range(len(features)):
                # If the current item label is the same as the one we are looking for
                if labels[j] == i:
                    actual_feature = features[j]
                    features_sum.append(actual_feature)

            # Already collected all the sum of features from class i
            # Get mean and std of columns, this should result in an 1 x 5 array
            class_means.append(np.array(features_sum).mean(axis=0))
            class_stds.append(np.array(features_sum).std(axis=0))

        # End of all loops for all the classes
        self.means = class_means
        self.stds = class_stds

    """"This function should receive a single value (array with features)"""
    def get_class_probs(self, x):
        class_probs = []
        # prob_sum = 0

        # Calculate probability for every class
        for c in range(self.n_classes):
            gauss_result = 0

            # Loop through all the features of the item
            for i in range(len(x)):
                # mean and std to use for each selected class for each feature
                mu = self.means[c][i]
                sigma = self.stds[c][i]

                # Multiplication for the gaussian
                gauss = (1.0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((x[i] - mu) / sigma)**2)

                # We take the log and turn it into a sum due to underflow issues with the probabilities
                if gauss > 0:
                    gauss_result += math.log(gauss)

            gauss_result += math.log(self.priors[c])
            class_probs.append(gauss_result)
            # prob_sum += math.log(self.priors[c])

        # Normalization step of probabilities
        # for i in range(len(class_probs)):
        #    class_probs[i] = class_probs[i] / prob_sum

        return class_probs

    def fit(self, x, y, epochs, batch_size):
        # Calculate the priors
        self.set_priors(y)

        # Calculate means and stds for distributions
        self.set_means_and_stds(x, y)

        self.get_class_probs(x[0])

    def predict(self, x):
        predictions = []

        # Get probabilities for a given x
        for item in x:
            probs = self.get_class_probs(item)
            predictions.append(np.argmax(probs))

        return predictions

    def evaluate(self, x, y, print_metrics=True):
        prediction = self.predict(x)
        accuracy, precision, recall, f1_score = self.get_test_metrics(prediction, y, print_metrics)

        return accuracy, precision, recall, f1_score
