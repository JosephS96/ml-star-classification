from classifiers.NeuralClassifier import NeuralClassifier
from PlotWrapper import PlotWrapper
from dataset.DatasetWrapper import DatasetWrapper
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


print("this if ML")

data = DatasetWrapper()
train_data, train_labels = data.get_training_data()

test_data, test_labels = data.get_testing_data()

metrics = classification_report(test_labels, test_labels, output_dict=True)

#print(tr)

#test, lbl = data.get_testing_data()

#print(test)

# fashion_mnist = keras.datasets.fashion_mnist

# (train_data, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

neural = NeuralClassifier()

history_train = neural.fit(train_data, train_labels, epochs=5, batch_size=10)
results = neural.evaluate(test_data, test_labels)

print()

plotter = PlotWrapper()
#plotter.plot([history_train.history['accuracy']], title='model accuracy', y_label='accuracy', x_label='epoch')


