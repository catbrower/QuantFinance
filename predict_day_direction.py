import os
import ray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from StockData import StockData, NormalizedIndicatorData, load_raw_data_from_multiple_csvs, calculate_statistics_for_day

# given some date, lookup in a df what the index_day would be
def date_to_index_day(dates, df):
    results = []
    for date in dates:

        indecies = df[df.index.date == date.date()]
        if len(indecies) == 0:
            results.append(None)
            continue

        indecies = pd.unique(indecies['index_day'])
        if len(indecies) > 1:
            raise Exception('Dataframe is malformed')
        results.append(indecies[0])

    return pd.DataFrame({'index_day': results, 'time': dates}, index=dates.index)

# for each index in index_days, select from the df rows within range
def select_df_ranges_from_dates(df, index_days, range):
    if range < 0:
        raise Exception('Range must be positive. Subtract from index_days if selecting lookback')
    return [
        (df[df['index_day'] >= i]) & (df[df['index_day'] < i + range])
        for i in index_days
    ]

# auto-arima for prediction
# https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

# ray start --head --port=6379
ray.init()

csv_files = [f'data/{x}' for x in os.listdir('data') if x.startswith('spy_stock_minute')]
spy_data = StockData(load_raw_data_from_multiple_csvs(csv_files))
spy_dividend = pd.read_csv('data/spy_dates_dividends.csv')
spy_dividend['ex_date'] = pd.to_datetime(spy_dividend['ex_date'], format='%m/%d/%Y')
index_day = date_to_index_day(spy_dividend['ex_date'], spy_data.data)
spy_dividend['index_day'] = index_day['index_day']
spy_dividend = spy_dividend.dropna()

# test = spy_data.compute_indicators_for_day(0, spy_data.default_indicator_spec)
indicators = spy_data.compute_indicators()
daily_stats = indicators.compute_daily_statistics()
normalized_indicators = NormalizedIndicatorData(indicators, daily_stats)

# Combine the results back into a single DataFrame
# processed_rows = pd.concat(results, ignore_index=True)

# Shutdown ray
# ray.shutdown()

i = 0

#NOTE in order to properly normalize, I'll have to compute the mean and STD for each indicator over some previous window
# using the moving average might not work... or will it? I really don't know right now... Either way, it would be useful
# to have some code to compute daily stats like the mean and std of some indicator over n previous days, so I should just
# code that as it will be useful at some point

day_data = results.get_data_for_day(1)
training_set = day_data.to_training_set()
# plt.plot(day_data['close'], c='k')
columns = []
# for i in range(60):
#     plt.plot(np.arange(390), results[results['index_day'] == i]['dwt'])
# plt.scatter(results['close_std'], results['close_avg'])
    # plt.plot(np.arange(390), results[results['index_day'] == i]['close_avg'])
    # plt.plot(np.arange(390), results[results['index_day'] == i]['close_std'])
# high / low oscillator, possible to make supports better?
# hl_osc = (day_data['pct'] - day_data['rolling_min']) / (day_data['rolling_max'] - day_data['rolling_min'])
    # plt.plot(np.arange(390), day_data['close'] -  day_data[f'vwap_{i}'])

plotData = day_data.indicator('vwap', day=1).data.T
# norm = Normalize(vmin=plotData.min(), vmax=plotData.max())
plt.imshow(plotData, norm=CenteredNorm(), cmap='RdYlGn')
# plt.plot(np.zeros(390))
# plt.plot(day_data['close'])
plt.show(block=True)
print('done')