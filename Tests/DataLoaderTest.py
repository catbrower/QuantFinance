from research.data_loader import *
from Indicators import *

indicators = [
    IndicatorMovingAverageConvergenceDivergence(30, 10, 20),
    IndicatorStandardDeviation(30)
]

data = load_data_dict(indicators)

print()