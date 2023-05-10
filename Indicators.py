import numpy as np
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import CCIIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

class Indicator:
    BB = 'bb'
    MACD = 'macd'
    STD = 'std'
    RSI = 'rsi'
    VWAP = 'vwap'
    ATR = 'atr'

    def __init__(self, name, signals):
        self.name = name
        self.signals = signals

    def num_signals(self):
        return len(self.signals)

    def calculate(self, data):
        raise Exception('Unimplemented')
    
    def get_signal_names(self):
        return [x for x in self.signals]
    
class IndicatorSimple(Indicator):
    def __init__(self, name, period):
        self.signal_name = f'%s_%d' % (name, period)
        super().__init__(name, {self.signal_name: period})

class IndicatorMultiSignal(Indicator):
    def __init__(self, name, signals):
        self.name = name
        self.signals = signals

class IndicatorStandardDeviation(IndicatorSimple):
    def __init__(self, period):
        super().__init__(Indicator.STD, period)

    def calculate(self, data, data_column_0='close'):
        return [
            {'name': self.signal_name, 'value': data[data_column_0].rolling(self.signals[self.signal_name]).apply(lambda x: np.std(x))}
        ]
    
class IndicatorRelativeStrengthIndex(IndicatorSimple):
    def __init__(self, period):
        super().__init__(Indicator.RSI, period)

    def calculate(self, data, data_column_0='close'):
        return [
            {'name': self.signal_name, 'value': RSIIndicator(data[data_column_0], window=self.signals[self.signal_name]).rsi()}
        ]
    
class IndicatorVolumeWeightedAveragePrice(IndicatorSimple):
    def __init__(self, period):
        super().__init__(Indicator.VWAP, period)

    def calculate(self, data, data_column_0='high', data_column_1='low', data_column_2='close', data_column_3='volume'):
        indicator = VolumeWeightedAveragePrice(
            high=data[data_column_0],
            low=data[data_column_1],
            close=data[data_column_2],
            volume=data[data_column_3],
            window=self.signals[self.signal_name])
        
        return [
            {'name': self.signal_name, 'value': indicator.volume_weighted_average_price()}
        ]

class IndicatorAverageTrueRange(IndicatorSimple):
    def __init__(self, period):
        super().__init__(IndicatorSimple.ATR, period)

    def calculate(self, data, data_column_0='high', data_column_1='low', data_column_2='close'):
        indicator = AverageTrueRange(
            high = data[data_column_0],
            low = data[data_column_1],
            close = data[data_column_2],
            window=self.signals[self.signal_name])
        
        return [
            {'name': self.signal_name, 'value': indicator.average_true_range()}
        ]
    
class IndicatorBollingerBands(IndicatorMultiSignal):
    HIGH_BAND = 'hband'
    LOW_BAND = 'lband'

    def __init__(self, period):
        self.upper_signal_name = f'%s_%s_%d' % (Indicator.BB, IndicatorBollingerBands.HIGH_BAND, period)
        self.lower_signal_name = f'%s_%s_%d' % (Indicator.BB, IndicatorBollingerBands.LOW_BAND, period)

        super().__init__(Indicator.BB, {
            self.upper_signal_name: period, 
            self.lower_signal_name: period
        })
        

    def calculate(self, data, data_column_0='close'):
        period = self.signals[self.upper_signal_name]
        bb = BollingerBands(data[data_column_0], period)
        return [
            {'name': self.upper_signal_name, 'value': bb.bollinger_hband_indicator()},
            {'name': self.lower_signal_name, 'value': bb.bollinger_lband_indicator()}
        ]
    
class IndicatorMovingAverageConvergenceDivergence(IndicatorMultiSignal):
    PERIOD_LONG = 'period_long'
    PERIOD_SHORT = 'period_short'
    PERIOD_SIGNAL = 'period_signal'

    def __init__(self, period_long, period_short, period_signal):
        self.signal_name = f'%s_%d_%d_%d' % (Indicator.MACD, period_long, period_short, period_signal)
        super().__init__(Indicator.MACD, {
            IndicatorMovingAverageConvergenceDivergence.PERIOD_LONG: period_long,
            IndicatorMovingAverageConvergenceDivergence.PERIOD_SHORT: period_short,
            IndicatorMovingAverageConvergenceDivergence.PERIOD_SIGNAL: period_signal
        })

    def get_signal_names(self):
        return [self.signal_name]

    def calculate(self, data, data_column_0='close'):
        macd = MACD(
            data[data_column_0],
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_LONG],
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_SHORT],
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_SIGNAL]
        ).macd()

        return [
            {'name': self.signal_name, 'value': macd}
        ]