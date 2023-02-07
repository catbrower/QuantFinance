import json
from datetime import datetime

import pandas as pd

data_file = open('data/eurusd.json')
data_json = json.load(data_file)['results']
data_raw = {
    'volume': [],
    'vw': [],
    'open': [],
    'high': [],
    'low': [],
    'close': [],
    't': [],
    'n': []
}

for row in data_json:
    data_raw['volume'].append(row['v'])
    data_raw['vw'].append(row['vw'])
    data_raw['open'].append(row['o'])
    data_raw['high'].append(row['h'])
    data_raw['low'].append(row['l'])
    data_raw['close'].append(row['c'])
    data_raw['t'].append(row['t'])
    data_raw['n'].append(row['n'])

data = pd.DataFrame(data_raw)
data['time'] = pd.to_datetime(data['t'], unit='ms')
# data = data.set_index(['time'])

print()