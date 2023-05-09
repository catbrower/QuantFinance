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

    def __init__(self, name, signal_columns):
        self.name = name
        self.signal_columns = signal_columns

    def num_signals(self):
        return len(self.signal_columns)

    def calculate(self, data):
        raise Exception('Unimplemented')
    
class IndicatorSimple(Indicator):
    def __init__(self, name, period):
        super().__init__(name, {'period': period})

class IndicatorMultiSignal(Indicator):
    def __init__(self, name, signals):
        self.name = name
        self.signals = signals

class IndicatorStandardDeviation(IndicatorSimple):
    def __init__(self, period):
        super().__init__(Indicator.STD, period)

    def calculate(self, data, data_column_0='close'):
        return data[data_column_0].rolling(self.signals['period']).apply(lambda x: np.std(x))
    
class IndicatorRelativeStrengthIndex(IndicatorSimple):
    def __init__(self, period):
        super().__init__(Indicator.RSI, period)

    def calculate(self, data, data_column_0='close'):
        return RSIIndicator(data[data_column_0], window=self.signals['period']).rsi()
    
class IndicatorVolumeWeightedAveragePrice(IndicatorSimple):
    def __init__(self, period):
        super().__init__(Indicator.VWAP, period)

    def calculate(self, data, data_column_0='high', data_column_1='low', data_column_2='close', data_column_3='volume'):
        indicator = VolumeWeightedAveragePrice(
            high=data[data_column_0],
            low=data[data_column_1],
            close=data[data_column_2],
            volume=data[data_column_3],
            window=self.signals['period'])
        
        return indicator.volume_weighted_average_price()

class IndicatorAverageTrueRange(IndicatorSimple):
    def __init__(self, period):
        super().__init__(IndicatorSimple.ATR, period)

    def calculate(self, data, data_column_0='high', data_column_1='low', data_column_2='close'):
        indicator = AverageTrueRange(
            high = data[data_column_0],
            low = data[data_column_1],
            close = data[data_column_2],
            window=self.signals['period'])
        
        return indicator.average_true_range()
    
class IndicatorBollingerBands(IndicatorMultiSignal):
    HIGH_BAND = 'hband'
    LOW_BAND = 'lband'

    def __init__(self, period):
        super().__init__(Indicator.BB, {
            IndicatorBollingerBands.HIGH_BAND: period, 
            IndicatorBollingerBands.LOW_BAND: period
        })

    def calculate(self, data, data_column_0='close'):
        bb = BollingerBands(data[data_column_0], self.period)
        return [
            {'name': f'%s_%s_%d' % (Indicator.BB, IndicatorBollingerBands.HIGH_BAND, self.signals['period']), 'value': bb.bollinger_hband_indicator()},
            {'name': f'%s_%s_%d' % (Indicator.BB, IndicatorBollingerBands.LOW_BAND, self.signals['period']), 'value': bb.bollinger_lband_indicator()}
        ]
    
class IndicatorMovingAverageConvergenceDivergence(IndicatorMultiSignal):
    PERIOD_LONG = 'period_long'
    PERIOD_SHORT = 'period_short'
    PERIOD_SIGNAL = 'period_signal'

    def __init__(self, period_long, period_short, period_signal):
        super().__init__(Indicator.MACD, {
            IndicatorMovingAverageConvergenceDivergence.PERIOD_LONG: period_long,
            IndicatorMovingAverageConvergenceDivergence.PERIOD_SHORT: period_short,
            IndicatorMovingAverageConvergenceDivergence.PERIOD_SIGNAL: period_signal
        })

    def calculate(self, data, data_column_0='close'):
        macd = MACD(
            data[data_column_0],
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_LONG],
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_SHORT],
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_SIGNAL]
        ).macd()
        
        signal_name = f'%s_%d_%d_%d' % (
            Indicator.MACD,
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_LONG],
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_SHORT],
            self.signals[IndicatorMovingAverageConvergenceDivergence.PERIOD_SIGNAL],
        )

        return [
            {'name': signal_name, 'value': macd.macd()}
        ]