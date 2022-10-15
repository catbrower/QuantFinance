import numpy as np
import pandas as pd
from tensorflow import keras
import keras.utils
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, LSTM

class RandomModel:
    def from_weights(weights, num_inputs):
        result = RandomModel(num_inputs)
        result.set_weights(weights)
        return result

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs

        # Build Model
        input = Input(shape=(None, num_inputs))
        lstm1 = LSTM(64, return_sequences=True)(input)
        lstm2 = LSTM(64, return_sequences=True)(lstm1)
        lstm3 = LSTM(64)(lstm2)
        output = Dense(1, activation='sigmoid')(lstm3)

        model = Model(inputs=input, outputs = output)
        # model.add(Dense(1))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def get_input_shape(self):
        return (None, self.num_inputs)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def randomize(self, scale):
        weights = self.model.get_weights()
        new_weights = []

        for layer in weights:
            randoms = np.random.normal(scale=scale, size=layer.shape)
            new_weights.append(randoms + layer)

        self.model.set_weights(new_weights)

    def plot(self):
        keras.utils.plot_model(self.model)

    def summary(self):
        return self.model.summary()

    def predict(self, inputs):
        return self.model.predict(inputs)