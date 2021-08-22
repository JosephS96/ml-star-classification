import numpy as np

class KFoldCrossValidation:
    def __init__(self, classifier, x, y, n_splits=4):
        self.classifier = classifier
        self.train_data = x
        self.train_labels = y
        self.folds = n_splits

        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []

    def get_training_scores(self):
        current_start = 0
        fold_size = self.get_fold_sizes()
        current_next = fold_size

        # train and evaluate folds
        for i in range(self.folds):
            train_data = self.train_data.copy()
            train_labels = self.train_labels.copy()

            # Separate test data
            test_data = train_data[current_start:current_next]
            test_labels = train_labels[current_start:current_next]

            # Remove test data from train data
            del train_data[current_start:current_next]
            del train_labels[current_start:current_next]

            # Update indexes
            current_start = current_next
            current_next += fold_size

            # Train classifier with current fold training data
            self.classifier.fit(train_data, train_labels, epochs=50, batch_size=10)

            # Evaluation of the model on the given fold
            accuracy, precision, recall, f1_score = self.classifier.evaluate(test_data, test_labels)

            # Save results
            self.accuracy.append(accuracy)
            self.precision.append(precision)
            self.recall.append(recall)
            self.f1_score.append(f1_score)

        return np.mean(self.accuracy), np.mean(self.precision), np.mean(self.recall), np.mean(self.f1_score)

    def final_evaluation(self, x, y):
        accuracy, precision, recall, f1_score = self.classifier.evaluate(x, y)
        return accuracy, precision, recall, f1_score

    def get_fold_sizes(self):
        n_size = round(len(self.train_labels) / self.folds)
        return n_size
