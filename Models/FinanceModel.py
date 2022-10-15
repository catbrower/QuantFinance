import os
import hashlib

from tensorflow import keras

from models import *
from data_loader import *
from Database import Database

class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, indicators):
        indicator_str = ''.join([f'%s%d' % (key, indicators[key]) for key in indicators])
        # m = hashlib.md5()
        # m.update(indicator_str)
        # self.indicator_digest = m.digest()
        self.db = Database()
        self.indicators = indicators

    def on_epoch_end(self, epoch, logs=None):
        db_client = self.db.get_instance()
        db_models = db_client['finance']['models']

        model = {
            'metrics': {
                'loss': logs.get('loss'),
                'precision': logs.get('precision'),
                'recall': logs.get('recall'),
                'accuracy': logs.get('accuracy')
            },
            'indicators': self.indicators,
            'model': [x.tolist() for x in self.model.get_weights()]
        }

        db_models.insert_one(model)
        
class FinanceModel():
    def __init__(self, dataset):
        self.dataset = dataset
        self.epochs = 10

        self.columns = [
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

        self.indicators = {
            'rsi_period': 30,
            'vwap_period': 46,
            'atr_period': 39,
            'macd_short': 21,
            'macd_long': 4,
            'macd_signal': 21,
            'bb_period': 4,
            'std': 33
        }

        self.hyperparams = {
            'threshold': {
                'value': 0.55, 'min': 0, 'max': 1, 'step': 0.01
            },
            'num_layers': {
                'value': 3, 'min': 1, 'max': 10, 'step': 1
            }
        }

        self.lookback = 30

        neg, pos = np.bincount(self.dataset['binary_reward_buy'])
        total = neg + pos
        self.initial_bias_buy = np.log([pos / neg])
        self.class_weight_buy = {
            0: (1 / neg) * (total / 2),
            1: (1 / pos) * (total / 2)
        }

        self.model = deep_dense(
            self.hyperparams,
            self.lookback,
            len(self.columns),
            steps_per_epoch=max(dataset['index_day']),
            output_bias=self.initial_bias_buy)

    def set_indicators(self, indicators):
        self.indicators = indicators

    def fit(self):
        steps_in_day = len(self.dataset[self.dataset['index_day'] == 0].dropna())
        lonest_indicator = 390 - steps_in_day
        self.history = self.model.fit(
            batch_train_generator_fixed(self.dataset, 
                self.lookback,
                self.columns,
                None,
                max(self.dataset['index_day'])),
            steps_per_epoch=max(self.dataset['index_day']),
            batch_size=steps_in_day,
            epochs=self.epochs,
            verbose=1,
            class_weight=self.class_weight_buy,
            callbacks=[SaveModelCallback(self.indicators)]
        )
        return self.history

    def save(self):
        self.model.save('trained.model')