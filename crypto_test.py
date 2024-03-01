# Intent is for this script to be run once a day by cron job
# fetch the last two days of ETH and BTC data, buy which ever
# has increased the most

import time
import requests
import functools
import pandas as pd
from datetime import datetime, timedelta

url_base = 'https://api.polygon.io/v2/aggs/ticker'
url_params = 'adjusted=true&sort=asc&limit=120&apiKey=PrCJ1R_Sa_jfqIzP_un7pjwsVcS_TTd5m_vGs1'
coins = {
    'ETH': {
        'polygon_symbol': 'X:ETHUSD',
        'data': None
    },
    'BTC': {
        'polygon_symbol': 'X:BTCUSD',
        'data': None
    }
}

error = None
date_to = datetime.fromtimestamp(time.time())
date_from = date_to - timedelta(days=2)
date_to = datetime.strftime(date_to, '%Y-%m-%d')
date_from = datetime.strftime(date_from, '%Y-%m-%d')

for key in coins:
    url = f'{url_base}/{coins[key]["polygon_symbol"]}/range/1/day/{date_from}/{date_to}?{url_params}'

    result = requests.get(url).json()
    if 'resultsCount' not in result or result['resultsCount'] < 1:
        print('Error fetching data, no results returned')
        error = 'No results returned'

    data = {}
    count = 0
    for row in result['results']:
        for row_key in row:
            if count == 0:
                data[row_key] = [row[row_key]]
            else:
                data[row_key].append(row[row_key])
        count += 1

    coins[key]['data'] = pd.DataFrame(data).set_index('t')

with open('output.txt', 'a') as file:
    if error is not None:
        file.write(f'{time.time()},{error}\n')
    else:
        # Determine trade position
        prices = []
        position = 'USDT'
        max_change = 0
        for key in coins:
            data = coins[key]['data']
            value_0 = data.iloc[0][['o', 'h', 'l', 'c']].mean()
            value_1 = data.iloc[-1][['o', 'h', 'l', 'c']].mean()
            prices.append((key, value_1))
            change = (value_1 - value_0) / value_0
            if change > max_change:
                max_change = change
                position = key
                
        file.write(f'{time.time()},{position},{prices}\n')
print()
