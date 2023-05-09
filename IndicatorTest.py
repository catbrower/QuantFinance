from research.Indicators import *

std = IndicatorStandardDeviation(30)
rsi = IndicatorRelativeStrengthIndex(30)
vwap = IndicatorVolumeWeightedAveragePrice(30)
atr = IndicatorAverageTrueRange(30)
bb = IndicatorBollingerBands(30)
macd = IndicatorMovingAverageConvergenceDivergence(30, 10, 20)

