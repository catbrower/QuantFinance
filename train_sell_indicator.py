import os

import numpy as np
from Models import RandomModel

from data_loader import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_inputs = 5
lookback = None

rsi_period = 5
vwap_period = 30
cci_period = 30
atr_period = 30
reward_period = 30
longest_indicator = 48
buy_threshold = 0.75

columns = [
    'pct_change',
    'rsi',
    'vwap',
    'macd',
    'atr'
]

data = load_data()
model = RandomModel.RandomModel(num_inputs).model

num_days = 248

# Prepare training set
def train_generator():
    while True:
        for index_day in range(num_days):
            selection = data[data['index_day'] == index_day]
            if len(selection) != 390:
                print(f'warning, data for day %d is malformed' % index_day)
                continue

            for index_minute in range(longest_indicator, 390):
                train_x = selection.iloc[longest_indicator:index_minute + 1][columns].values.tolist()
                train_y = float(selection.iloc[index_minute]['binary_reward'])
                yield np.array([train_x]), np.array([train_y])

model.fit_generator(train_generator(), steps_per_epoch=390 - longest_indicator, epochs=num_days, verbose=1)
model.save('sell_indicator.model')
