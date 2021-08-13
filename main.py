from classifiers.KnnClassifier import KnnClassifier
from classifiers.NeuralClassifier import NeuralClassifier
from PlotWrapper import PlotWrapper
from dataset.DatasetWrapper import DatasetWrapper
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

print("this if ML")

data = DatasetWrapper()
train_data, train_labels = data.get_training_data()

test_data, test_labels = data.get_testing_data()

#knn = KnnClassifier()
#knn.fit(train_data, train_labels, epochs=5, batch_size=0)

#predictions = knn.predict(test_data)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data, train_labels)

predictions = knn.predict(test_data)

print(classification_report(test_labels, predictions))

#print(tr)

#test, lbl = data.get_testing_data()

#print(test)

# fashion_mnist = keras.datasets.fashion_mnist

# (train_data, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

neural = NeuralClassifier()

#history_train = neural.fit(train_data, train_labels, epochs=5, batch_size=10)
#accuracy, precision, recall, f1_score = neural.evaluate(test_data, test_labels)

#print()

#plotter = PlotWrapper()
#plotter.plot([history_train.history['accuracy']], title='model accuracy', y_label='accuracy', x_label='epoch')


