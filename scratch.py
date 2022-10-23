from Database import Database
from Models.Tuner import Tuner
from Models.HyperParameter import HyperParameter

indicators = {
    'rsi_period': HyperParameter(2, 60, 1, value=30),
    'vwap_period': HyperParameter(2, 60, 1, value=46),
    'atr_period': HyperParameter(2, 60, 1, value=39),
    'macd_short': HyperParameter(2, 60, 1, value=21),
    'macd_long': HyperParameter(2, 60, 1, value=4),
    'macd_signal': HyperParameter(2, 60, 1, value=21),
    'bb_period': HyperParameter(2, 60, 1, value=4),
    'std': HyperParameter(2, 60, 1, value=33)
}

tuner = Tuner([x for x in indicators], 'binary_reward_buy', indicators)
tuner.search(1)