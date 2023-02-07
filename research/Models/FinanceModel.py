from tensorflow import keras

from models import *
from data_loader import *
from Database import Database
import Models.HyperParameter as HyperParameter

class SaveModelCallback(keras.callbacks.Callback):
    def __init__(self, indicators, x_columns, y_columns, lookback):
        # indicator_str = ''.join([f'%s%d' % (key, indicators[key]) for key in indicators])
        # m = hashlib.md5()
        # m.update(indicator_str)
        # self.indicator_digest = m.digest()
        self.db = Database()
        _indicators = {}
        for key in indicators:
            _indicators[key] = indicators[key].value
        self.indicators = _indicators
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.lookback = lookback

    def on_epoch_end(self, epoch, logs=None):
        db_client = self.db.get_instance()
        db_models = db_client['finance']['models']

        model = {
            'version': 2,
            'metrics': {
                'loss': logs.get('loss'),
                'precision': logs.get('precision'),
                'recall': logs.get('recall'),
                'accuracy': logs.get('accuracy')
            },
            'epoch': epoch,
            'x_columns': self.x_columns,
            'y_columns': self.y_columns,
            'lookback': self.lookback,
            'indicators': self.indicators,
            'model': [x.tolist() for x in self.model.get_weights()]
        }

        db_models.insert_one(model)
        
# X represents model inputs
# Y represents what the model is predicting
class FinanceModel():
    def from_weights(weights, indicators, Y, dataset):
        result = FinanceModel(indicators, Y, dataset)
        result.model.set_weights(weights)
        return result

    def __init__(self, indicators, Y, dataset):
        self.dataset = dataset
        self.epochs = 10
        self.indicators = indicators
        self.X = self.get_x_columns()
        self.Y = Y
        self.y_columns = Y
        self.steps_in_day = len(self.dataset[self.dataset['index_day'] == 0].dropna())

        # self.indicators = {
        #     'rsi_period': HyperParameter(2, 60, 1, value=30),
        #     'vwap_period': HyperParameter(2, 60, 1, value=46),
        #     'atr_period': HyperParameter(2, 60, 1, value=39),
        #     'macd_short': HyperParameter(2, 60, 1, value=21),
        #     'macd_long': HyperParameter(2, 60, 1, value=4),
        #     'macd_signal': HyperParameter(2, 60, 1, value=21),
        #     'bb_period': HyperParameter(2, 60, 1, value=4),
        #     'std': HyperParameter(2, 60, 1, value=33)
        # }

        self.hyperparams = {
            'threshold': HyperParameter.HyperParameter(0, 1, 0.01, value=0.55),
            'num_layers': HyperParameter.HyperParameter(1, 10, 1, value=3)
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
            len(self.X),
            steps_per_epoch=max(dataset['index_day']),
            output_bias=self.initial_bias_buy)

    def randomize_weights(self, scale):
        pass

    # Hard coded for now
    def get_x_columns(self):
        return [
            'rsi',
            'vwap',
            'atr',
            'macd',
            'bb_hband',
            'bb_lband',
            'std'
        ]

    def predict(self):
        predictions = self.model.predict(
            batch_train_generator_fixed(
                self.dataset,
                self.lookback,
                self.X,
                self.Y,
                self.steps_in_day,
                max_batches=max(self.dataset['index_day'])),
            batch_size=self.steps_in_day,
            verbose=1)
            # callbacks=[SaveModelCallback(self.indicators)])
        return predictions

    def fit(self):
        self.history = self.model.fit(
            batch_train_generator_fixed(
                self.dataset, 
                self.lookback,
                self.X,
                self.Y,
                max(self.dataset['index_day'])),
            steps_per_epoch=max(self.dataset['index_day']),
            batch_size=self.steps_in_day,
            epochs=self.epochs,
            verbose=1,
            class_weight=self.class_weight_buy,
            callbacks=[SaveModelCallback(self.indicators, self.get_x_columns(), self.y_columns, self.lookback)]
        )
        return self.history

    def save(self):
        self.model.save('trained.model')