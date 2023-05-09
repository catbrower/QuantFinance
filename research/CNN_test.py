import numpy as np
import pandas as pd
from data_loader import *

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from research.Indicators import IndicatorsFactory 

columns = [
    # 'pct_mean',
    'volume',
    'bb_hband',
    'bb_lband',
    'rsi',
    'vwap',
    'macd',
    'atr',
    'std'
]

def create_training_sets():
    indicators = [
        IndicatorsFactory.standard_deviation(30),
        IndicatorsFactory.relative_strength_index(30),
        IndicatorsFactory.volume_weighted_average_price(30),
        IndicatorsFactory.average_true_range(30),
        IndicatorsFactory.bollinger_bands(30),
        IndicatorsFactory.moving_average_convergence_divergence(30, 15, 22)
    ]

    num_inputs = sum([len(x['signal_columns']) for x in indicators])

    total_days = 248
    num_days = 248

    data = load_data_dict(indicators)



train_X, train_Y, text_X, test_Y = create_training_sets()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print()