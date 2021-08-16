from classifiers.KnnClassifier import KnnClassifier
from classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from classifiers.NeuralClassifier import NeuralClassifier
from PlotWrapper import PlotWrapper
from dataset.DatasetWrapper import DatasetWrapper
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import numpy as np

print("this if ML")

# Defines train and test data for classifiers
data = DatasetWrapper()
train_data, train_labels = data.get_training_data()
test_data, test_labels = data.get_testing_data()

bayes = NaiveBayesClassifier(n_classes=6)
bayes.fit(train_data, train_labels, epochs=5, batch_size=0)

# bayes = GaussianNB()
# bayes.fit(train_data, train_labels)

# predictions = bayes.predict(test_data)

bayes.evaluate(test_data, test_labels)

# print(classification_report(test_labels, predictions))

#plotter = PlotWrapper()
#plotter.plot([history_train.history['accuracy']], title='model accuracy', y_label='accuracy', x_label='epoch')


