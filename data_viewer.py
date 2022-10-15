import numpy as np
from numpy_fracdiff import fracdiff
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from data_loader import *
from math_util import *

data = load_data()

columns = [
    'pct_close',
    'rsi',
    'vwap',
    'macd',
    'atr'
]
reward_period = 7
index_day = 100
longest_indicator = 48
selection = data[data['index_day'] == index_day]

model_buy = keras.models.load_model('test_buy.model')
model_sell = keras.models.load_model('indicator_sell.model')

# Do predicts
# predicts_buy = model_buy.predict(data_generator_fixed(data, 30, 1, columns, longest_indicator))
# predicts_buy = [1 if x > 0.5 else 0 for x in predicts_buy]

mean = data[['open', 'high', 'low', 'close']].mean(axis=1)
mean = mean - mean.shift(1)

high = data['high'] - data['high'].shift(1)
low = data['low'] - data['low'].shift(1)
close = data['close'] - data['close'].shift(1)

# fd_mean = fractional_difference(selection['pct_close'], .5, cutoff=.05)
# fd_mean = fd_mean - fd_mean.dropna()[0]
# fd_high = fd_mean + high - mean
# fd_low = fd_mean + low - mean
# fd_close = fd_mean + close - mean

# print(len(selection) - len(fd_mean))
# test = adfuller(fd_mean)
# print(test)
# print(test.pvalue)

mean_ask = selection[['askopen', 'askhigh', 'asklow', 'askclose']].mean(axis=1)
mean_bid = selection[['bidopen', 'bidhigh', 'bidlow', 'bidclose']].mean(axis=1)
mean_spread = mean_ask - mean_bid
mean_spread = mean_spread - mean_spread.mean()
mean_spread = mean_spread / (mean_spread.std() * 10)

volume = selection['volume'][0:-1]
volume = volume - volume.mean()
volume = volume / (volume.std() * 10)

print(selection['mean_spread'].mean())

# Calculate daily statistics
means = []
stds = []
for i in range(int(max(data['index_day']))):
    day = data[data['index_day'] == i]
    means.append(day['mean'].mean())
    stds.append(day['mean'].std())

ups = [x[0] for x in np.argwhere(np.array(means) > 0)]

# plt.scatter(means, stds)
plt.plot(data[data['index_day'].isin(ups)]['raw_mean'])
# plt.plot(mean_bid)
# plt.plot(data['close'])
plt.show(block=True)
print()