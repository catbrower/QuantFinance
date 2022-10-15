import ray
import numpy as np
import keras_tuner as kt

from data_loader import *
from models import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_threads = 16
lookback = 30
epochs = 5
longest_indicator = 48
buy_threshold = 0.75
steps_in_day = 390 - longest_indicator - lookback

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

num_inputs = len(columns)
total_days = 248
num_days = 248

hyperparams = {
    'threshold': {
        'value': 0.55, 'min': 0, 'max': 1, 'step': 0.01
    },
    'num_layers': {
        'value': 3, 'min': 1, 'max': 10, 'step': 1
    }
}

def generate_indicators(indicators):
    result = {}
    for key in indicators:
        result[key] = int(indicators[key] + np.abs(np.random.normal() * 2) + 1)

    return result

def train_model_buy(indicators):
    data = load_data(indicators=indicators)
    model = deep_dense(lookback, num_inputs)
    history = model.fit(train_generator_fixed(data, lookback), steps_per_epoch=steps_in_day * num_days, epochs=epochs, verbose=0)
    model.save('indicator_buy.model')
    return (indicators, history.history['accuracy'][-1])

def train_model_sell(indicators):
    data = load_data(indicators=indicators)
    model = deep_dense(lookback, num_inputs)
    model.save('indicator_sell.model')
    history = model.fit(train_generator_fixed(data, lookback, is_buy_indicator=False), steps_per_epoch=steps_in_day * num_days, epochs=epochs, verbose=0)
    return (indicators, history.history['accuracy'][-1])

@ray.remote
def train_model_buy_ray(indicators):
    return train_model_buy(indicators)

@ray.remote
def train_model_sell_ray(indicators):
    return train_model_sell(indicators)

# results = [train_model_buy_ray.remote(default_indicators), train_model_sell_ray.remote(default_indicators)]
# ray.get(results)
# print('done')

data = load_data()
# Calculate daily statistics
means = []
stds = []
for i in range(int(max(data['index_day']))):
    day = data[data['index_day'] == i]
    means.append(day['mean'].mean())
    stds.append(day['mean'].std())

ups = [x[0] for x in np.argwhere(np.array(means) > 0)]

neg, pos = np.bincount(data['binary_reward_buy'])
total = neg + pos
initial_bias_buy = np.log([pos / neg])
class_weight_buy = {
    0: (1 / neg) * (total / 2),
    1: (1 / pos) * (total / 2)
}

# prep train + test sets
# X = []
# Y = []
# for xy in train_generator_fixed(data, lookback):
#     X.append(xy[0])
#     Y.append(xy[1])

# loss: 0.6729 - accuracy: 0.6450 - recall: 0.4405 

# TODO it might be the case that the indicators I'm using can only find one signal and many are present.
# I might be able to correct for this by removing some of the positive rewards i.e setting binary_reward
# to zero in some places and see if this increases the accuracy. If it does, that supports my hypothesis
# Also it would be difficult to do this over many days at once so, start with 1 day and see how it goes

# We could also do the randomization thing and count who can pick the most correct points
# 1 point for correct, no loss for false negative, minus 1 for false positive

# Note: If using epochs make sure that the data is an integer multiple of epochs
model = deep_dense(
    hyperparams,
    lookback,
    num_inputs,
    steps_per_epoch=num_days,
    output_bias=initial_bias_buy)
keras.utils.plot_model(model, show_shapes=True, to_file='model.png')

history = model.fit(
    batch_train_generator_fixed(data, lookback, columns, None, num_days),
    steps_per_epoch=num_days,
    batch_size=steps_in_day,
    epochs=epochs,
    verbose=1,
    class_weight=class_weight_buy
)

model.save('trained.model')