from sklearn.tree import DecisionTreeClassifier

from classifiers.AdaBoostClassifier import AdaBoostClassifier
from classifiers.KnnClassifier import KnnClassifier
from classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from classifiers.NeuralClassifier import NeuralClassifier
from PlotWrapper import PlotWrapper
from dataset.DatasetWrapper import DatasetWrapper
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_gaussian_quantiles

import numpy as np

print("this if ML")

# Defines train and test data for classifiers
data = DatasetWrapper()
train_data, train_labels = data.get_training_data()
test_data, test_labels = data.get_testing_data()

x, y = make_gaussian_quantiles(n_samples=1300, n_features=6, n_classes=6, random_state=1)

boost = AdaBoostClassifier()
accuracy, precision, recall, f1_score = boost.fit(train_data, train_labels, epochs=100, batch_size=0)
# accuracy, precision, recall, f1_score = boost.fit(x.tolist(), y.tolist(), epochs=300, batch_size=0)
boost.evaluate(test_data, test_labels)

# bayes = NaiveBayesClassifier()
# bayes.fit(train_data, train_labels, epochs=0, batch_size=0)
# bayes.evaluate(test_data, test_labels)

#knn = KnnClassifier()
#knn.fit(train_data, train_labels, epochs=0, batch_size=0)

#knn.evaluate(test_data, test_labels)

# print(classification_report(test_labels, predictions))

plotter = PlotWrapper()
# plotter.plot([history_train.history['accuracy']], title='model accuracy', y_label='accuracy', x_label='epoch')
plotter.plot([accuracy], title='model accuracy', y_label='accuracy', x_label='epoch')


