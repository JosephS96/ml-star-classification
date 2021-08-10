import tensorflow as tf
from tensorflow import keras


class NeuralClassifier:
    def __init__(self):
        self.learning_rate = 0.001
        self.model = self.create_model()

    def create_model(self):
        model = keras.Sequential([
            # keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(6, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.SGD()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', 'mse'])
        return model

    def fit(self, x, y):
        history = self.model.fit(x, y, epochs=200, batch_size=10)
        return history

