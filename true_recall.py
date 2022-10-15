from json import load
import tensorflow as tf
from tensorflow import keras

from data_loader import *

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

model = keras.models.load_model('trained.model')

data = load_data()
lookback = 30

num_days = max(data['index_day'])
predictions = model.predict(
    batch_train_generator_fixed(data, lookback, columns, None, num_days)
)

print('done')