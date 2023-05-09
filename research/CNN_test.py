import numpy as np
import pandas as pd
from data_loader import *

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

def build_indicator_simple(name, period):
    return {
        'name': name,
        'period': period
    }

def build_indicator_macd(period_long, period_short, period_signal):
    return {
        'name': 'macd',
        'period_long': period_long,
        'period_short': period_short,
        'period_signal': period_signal
    }

def create_training_sets():
    num_inputs = len(columns)
    total_days = 248
    num_days = 248

    data = load_data_dict()



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