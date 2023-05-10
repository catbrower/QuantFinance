import numpy as np
import pandas as pd
from research.data_loader import *

from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tcn import TCN

from Indicators import *
from Util import flatten_list

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

lookback = 30
# train_X, train_Y, test_X, test_Y = create_training_sets()
train_X = np.load('data/tcn_trainx.npy')
train_Y = np.load('data/tcn_trainy.npy')
test_X = np.load('data/tcn_testx.npy')
test_Y = np.load('data/tcn_testy.npy')
# train_X, train_Y, test_X, test_Y = create_fake_training_sets(lookback)
shape = train_X[0].shape
model = Sequential([
    Input(shape=(1, 30, 7)),
    Conv2D(16, 5),
    MaxPool2D(),
    TCN(input_shape=shape,
        kernel_size=3,
        batch_size=len(train_X),
        use_skip_connections=False,
        use_batch_norm=False,
        use_weight_norm=False,
        use_layer_norm=False
        ),
    Dense(1, activation='tanh')
])
model.summary()
model.compile('adam', 'categorical_crossentropy')

model.fit(train_X, train_Y, epochs=100, verbose=2)

print()