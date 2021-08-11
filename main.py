from classifiers.NeuralClassifier import NeuralClassifier
from PlotWrapper import PlotWrapper
from dataset.DatasetWrapper import DatasetWrapper
import matplotlib.pyplot as plt

print("this if ML")

data = DatasetWrapper()
train_data, train_labels = data.get_training_data()

test_data, test_labels = data.get_testing_data()

#print(tr)

#test, lbl = data.get_testing_data()

#print(test)

# fashion_mnist = keras.datasets.fashion_mnist

# (train_data, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

neural = NeuralClassifier()
neural2 = NeuralClassifier()

history_train = neural.fit(train_data, train_labels, epochs=500, batch_size=10)
history_train2 = neural2.fit(train_data, train_labels, epochs=500, batch_size=20)
results = neural.evaluate(test_data, test_labels)

print(results)

plotter = PlotWrapper()
plotter.plot([history_train.history['accuracy'], history_train2.history['accuracy']], title='model accuracy', y_label='accuracy', x_label='epoch')

# summarize history for loss
plt.plot(history_train.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()

