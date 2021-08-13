from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np


class ModelHistory:
    def __init__(self, show_progress=True):
        self.n_epochs = 1

        #Global metrics of the model
        self.__accuracy = []
        self.__f1_score = []
        self.__precision = []
        self.__recall = []

        self.__show_progress = show_progress

    """This method should be called at the end of each epoch in order to calculate the metrics for the classifier at the specified epoch
        :arg predicted: array of predictions of the class
        :arg ground_truth array with the correct label corresponding to the predicted values
    """
    def save_metrics(self, predicted, ground_truth):
        metrics = classification_report(ground_truth, predicted, output_dict=True)

        # Calculate accuracy
        # self.__accuracy.append(self.calculate_accuracy(predicted, ground_truth))
        self.__accuracy.append(metrics['accuracy'])

        avg_metrics = metrics['macro avg']
        #avg_metrics = metrics['weighted avg']

        # Calculate precision
        self.__precision.append(avg_metrics['precision'])

        # Calculate recall
        self.__recall.append(avg_metrics['recall'])

        # Calculate f1-score
        self.__f1_score.append(avg_metrics['f1-score'])

        if self.__show_progress:
            self.print_epoch_metrics()

    """Method mainly used for getting the testing results printed
        Return the metrics of the evaluation of the global model
        also prints the whole report for per class metrics
    """
    def get_evaluation_report(self, predicted, ground_truth, print_report=True):

        if print_report:
            print(classification_report(ground_truth, predicted))

        metrics = classification_report(ground_truth, predicted, output_dict=True)
        accuracy = metrics['accuracy']

        avg_metrics = metrics['macro avg']
        # avg_metrics = metrics['weighted avg']

        precision = avg_metrics['precision']
        recall = avg_metrics['recall']
        f1_score = avg_metrics['f1-score']

        return accuracy, precision, recall, f1_score

    def get_training_metrics(self):
        return self.__accuracy, self.__precision, self.__recall, self.__f1_score

    def print_epoch_metrics(self):
        print(f"Epoch {len(self.__accuracy)}/{self.n_epochs} \n")
        print(
            f"[==============] - accuracy: {self.__accuracy[-1]} - f1 score: {self.__f1_score[-1]}")

    def set_n_epochs(self, epochs):
        self.n_epochs = epochs

    @property
    def show_progress(self):
        return self.__show_progress

    @show_progress.setter
    def show_progress(self, value):
        self.__show_progress = value

    """
    def calculate_accuracy(self, predicted, ground_truth):
        acc_counter = 0
        for i in range(len(predicted)):
            if predicted[i] == ground_truth[i]:
                acc_counter += 1
        accuracy = acc_counter / len(predicted)
        return accuracy

    def calculate_precision(self, predicted, ground_truth):
        true_positives = 0 # samples that were correctly classified as their corresponding class
        false_positives = 0 # wrongly classified samples, anything that is incorrect becomes false positive

        for i in range(len(predicted)):
            if predicted[i] == ground_truth[i]:
                true_positives += 1
            else:
                false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        return precision

    def calculate_recall(self, predicted, ground_truth):
        true_positives = 0  # samples that were correctly classified as their corresponding class
        false_negatives = 0  # samples that were predicted to a different class

        for i in range(len(predicted)):
            if predicted[i] == ground_truth[i]:
                true_positives += 1
            else:
                false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        return precision

    def calculate_f_score(self, predicted, ground_truth, weight = 1): """
