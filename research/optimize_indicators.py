import logging, os

import ray
import numpy as np
import keras_tuner as kt

from data_loader import *
from models import *

# Disable tensorflow logs because they're excessive and annoying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_threads = 16
lookback = 30
epochs = 10
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
        'value': 0.5, 'min': 0, 'max': 1, 'step': 0.01
    },
    'num_layers': {
        'value': 3, 'min': 1, 'max': 10, 'step': 1
    }
}

def worker(X, Y):
    lookback = 30
    epochs = 10
    indicators = generate_indicators()
    data = load_data(indicators=indicators)
    num_inputs = len(X)
    num_days = max(data['index_day'])

    # Calculate initial bias and class weights
    neg, pos = np.bincount(data['binary_reward_buy'])
    total = neg + pos
    initial_bias_buy = np.log([pos / neg])
    class_weight_buy = {
        0: (1 / neg) * (total / 2),
        1: (1 / pos) * (total / 2)
    }

    model = deep_dense(
        default_hyperparams,
        lookback,
        num_inputs,
        output_bias=initial_bias_buy)
    try:
        history = model.fit(
            batch_train_generator_fixed(data, lookback, X, Y, batch_size=num_days),
            steps_per_epoch=num_days,
            batch_size=steps_in_day,
            epochs=epochs,
            verbose=0,
            class_weight=class_weight_buy)

        return (indicators, history.history)
    except:
        return (indicators, None)

@ray.remote
def ray_worker(X, Y):
    return worker(X, Y)

@ray.remote
def ray_run_model(X, Y):
    result = worker(X, Y)
    return result

def generate_indicators():
    result = {}
    for key in default_indicators:
        result[key] = int(np.random.rand() * 59) + 2

    return result

# result = ray.get(ray_worker.remote(columns, 'binary_reward_buy'))
# result = ray.get(ray_run_model.remote(columns, 'binary_reward_buy'))
# print(result)
# quit()

for _ in range(100):
    try:
        jobs = []
        for _ in range(num_threads):
            jobs.append(ray_worker.remote(columns, 'binary_reward_buy'))

        results = ray.get(jobs)

        na_results = [x for x in results if x[1] is not None]

        if len(na_results) == 0:
            continue
        
        best_index = np.argmax([max(x[1]['precision']) for x in na_results])
        best_indicators = na_results[best_index][0]
        best_value = max(na_results[best_index][1]['precision'])

        print(best_value)
        print(best_indicators)
        print('-' * 50)
    except:
        pass
# result = worker(columns, 'binary_reward_buy')
# print(result)