from KFoldCrossValidation import KFoldCrossValidation
from classifiers.KnnClassifier import KnnClassifier
from dataset.DatasetWrapper import DatasetWrapper
from sklearn.preprocessing import StandardScaler
import pandas as pd

import numpy as np

print("this if ML")

# Defines train and test data for classifiers
data = DatasetWrapper()
train_data, train_labels = data.get_training_data()
test_data, test_labels = data.get_testing_data()

df = pd.DataFrame(train_data)
print(df.describe())

# Preprocessing and normalization of data
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data).tolist()
test_data = scaler.transform(test_data).tolist()

df2 = pd.DataFrame(train_data)
print(df2.describe())

cross_validation = False

# Create classifier
classifier = KnnClassifier(n_neighbors=5)
# classifier = NaiveBayesClassifier()
# classifier = AdaBoostClassifier()
# classifier = NeuralClassifier()

if cross_validation:
    kfold = KFoldCrossValidation(classifier, train_data, train_labels, n_splits=5)
    accuracy, precision, recall, f1_score = kfold.get_training_scores()

    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}".format(f1_score))

    print("Final evaluation")
    accuracy, precision, recall, f1_score = kfold.final_evaluation(test_data, test_labels)
else:
    accuracy, precision, recall, f1_score = classifier.fit(train_data, train_labels, epochs=100, batch_size=10)
    # classifier.fit(train_data, train_labels, epochs=100, batch_size=10)
    # history_train = classifier.fit(train_data, train_labels, epochs=100, batch_size=10)
    classifier.evaluate(test_data, test_labels)

# plotter = PlotWrapper()
# plotter.plot([history_train.history['accuracy']], title='model accuracy', y_label='accuracy', x_label='epoch')
# nnplotter.plot([accuracy], title='model accuracy', y_label='accuracy', x_label='epoch')


