import numpy as np
import pandas as pd
from research.data_loader import *

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Reshape
from keras.metrics import BinaryAccuracy, Recall, Precision

from tcn import TCN

from Indicators import *
from Util import flatten_list, calculate_class_weights

# returns train_X, train_Y, test_X, test_Y
def create_training_sets(lookback=30, train_split = 0.75):
    indicators = [
        IndicatorStandardDeviation(30),
        IndicatorRelativeStrengthIndex(30),
        IndicatorVolumeWeightedAveragePrice(30),
        IndicatorAverageTrueRange(30),
        IndicatorBollingerBands(30),
        IndicatorMovingAverageConvergenceDivergence(30, 15, 22)
    ]

    x_columns = flatten_list([x.get_signal_names() for x in indicators])
    y_columns = ['reward']

    data = load_data_dict(indicators).dropna()
    
    # Gather samples
    samples = []
    for index_day in range(max(data['index_day'])):
        for index_minute in range(int(min(data['index_minute'])), int(max(data['index_minute']) - lookback)):
            samples.append(data[
                (data['index_day'] == index_day) &
                (data['index_minute'] > index_minute) &
                (data['index_minute'] <= index_minute + lookback)
                ])

    split_index = int(len(samples) * train_split)
    train = samples[0:split_index]
    test = samples[split_index:]

    train_X = np.array([x[x_columns].values for x in train])
    train_Y = np.array([x[y_columns].values[-1] for x in train])
    test_X = np.array([x[x_columns].values for x in test])
    test_Y = np.array([x[y_columns].values[-1] for x in test])
    return train_X, train_Y, test_X, test_Y

# Just used to test that the ML model is setup correctly
def create_fake_training_sets(lookback, train_split=0.75):
    num_features = 10
    data_length = 100
    indicators = []

    data = pd.DataFrame({'reward': np.random.choice([-1, 0, 1], size=data_length)})
    for i in range(num_features):
        indicator_name = f'column_%d' % i
        indicators.append(indicator_name)
        data[indicator_name] = np.random.normal(size=data_length)

    # Gather training / testing samples
    samples = []
    for i in range(data_length - lookback):
        samples.append(data.iloc[range(i, i + lookback)])

    split_index = int(len(samples) * train_split)
    train = samples[0:split_index]
    test = samples[split_index:]

    train_X = np.array([x[indicators].values for x in train])
    train_Y = np.array([x['reward'].values[-1] for x in train])
    test_X = np.array([x[indicators].values for x in test])
    test_Y = np.array([x['reward'].values[-1] for x in test])
    return train_X, train_Y, test_X, test_Y

def reward_to_category(y):
    result = []
    for value in y:
        if value[0] == -1:
            result.append([1, 0])
        elif value[0] == 0:
            result.append([1, 0])
        else:
            result.append([0, 1])
    return np.array(result)

def build_model(input_shape, num_outputs):
    # (30, 7, 1)
    inputs = Input(shape=input_shape)
    x = Conv2D(16, 5)(inputs)
    x = MaxPool2D()(x)
    x = Reshape((1, np.prod(x.shape[1:])))(x)
    x = TCN(16)(x)
    x = Dense(num_outputs, activation='sigmoid')(x)

    return Model(inputs=[inputs], outputs=[x])

lookback = 30
# train_X, train_Y, test_X, test_Y = create_training_sets()
train_X = np.expand_dims(np.load('data/tcn_trainx.npy'), axis=3)
train_Y = np.array([1 if x == 1 else 0 for x in np.load('data/tcn_trainy.npy')])
test_X = np.load('data/tcn_testx.npy')
test_Y = np.load('data/tcn_testy.npy')
# train_X, train_Y, test_X, test_Y = create_fake_training_sets(lookback)
# for the input shape: (n_images, x_shape, y_shape, channels)
# this should be (1, lookback, # indicators, 1)

# TODO setup for custom loss in order to maximize recall
# https://stackoverflow.com/questions/52695913/custom-loss-function-in-keras-to-penalize-false-negatives

# TODO use clusting to label days

class_weight = calculate_class_weights(train_Y)
model = build_model((lookback, 7, 1), 1)
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[BinaryAccuracy(threshold=0.5), Recall(), Precision()])

model.fit(
    train_X,
    train_Y,
    class_weight=class_weight,
    epochs=100)

print()