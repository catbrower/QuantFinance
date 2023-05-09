import numpy as np
import pandas as pd
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

def generate_indicators():
    result = {}
    for key in default_indicators:
        result[key] = int(np.random.rand() * 59) + 2

    return result

num_inputs = len(columns)
total_days = 248
num_days = 248

indicators = generate_indicators()
data = load_data()

print()