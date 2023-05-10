from Indicators import *
from Util import generate_synthetic_ohlcv_data

# Just make sure everything runs without error

data = generate_synthetic_ohlcv_data(100, 100)

std = IndicatorStandardDeviation(30)
rsi = IndicatorRelativeStrengthIndex(30)
vwap = IndicatorVolumeWeightedAveragePrice(30)
atr = IndicatorAverageTrueRange(30)
bb = IndicatorBollingerBands(30)
macd = IndicatorMovingAverageConvergenceDivergence(30, 10, 20)

std.calculate(data)
rsi.calculate(data)
vwap.calculate(data)
atr.calculate(data)
bb.calculate(data)
macd.calculate(data)

