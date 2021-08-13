import tensorflow as tf
from tensorflow import keras
from classifiers.BaseClassifier import BaseClassifier
import numpy as np


class NeuralClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(6, activation='softmax')
        ])

        """Check later why it is not learning anything with SDG
        and how to adjust the parameters of the Adam optimizer
        How Adam works"""
        # optimizer = tf.keras.optimizers.SGD()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def fit(self, x, y, epochs, batch_size):
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
        return history

    def predict(self, x):
        output = self.model.predict(x)
        output = np.argmax(output, axis=1)

        return output

    def evaluate(self, x, y):
        #history = self.model.evaluate(x, y)
        #print(history)

        prediction = self.predict(x)
        accuracy, precision, recall, f1_score = self.get_test_metrics(prediction, y)

        return accuracy, precision, recall, f1_score


