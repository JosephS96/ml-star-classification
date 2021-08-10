from NeuralClassifier import NeuralClassifier
from dataset.DatasetWrapper import DatasetWrapper
import matplotlib.pyplot as plt

from tensorflow import keras

print("this if ML")

data = DatasetWrapper()
train_data, train_labels = data.get_training_data()

#print(tr)

#test, lbl = data.get_testing_data()

#print(test)

# fashion_mnist = keras.datasets.fashion_mnist

# (train_data, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

neural = NeuralClassifier()
history = neural.fit(train_data, train_labels)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()

