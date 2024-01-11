import os
import ray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

from StockData import StockData, NormalizedIndicatorData
from DataFrameUtil import load_raw_data_from_multiple_csvs, calculate_statistics

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
    result = [
        df[(df['index_day'] >= i) & (df['index_day'] < i + range)]
        for i in index_days
    ]
    return pd.concat(result)

# looking to see if there's a price anomaly related to the ex dividend date
# keeping all the code in here for cleanliness
# Note: it looks like the dividend strat will work.
# what needs to be done is find stocks that are overall doing well i.e are going up on the long & med term
# then check that they have high dividend. This should result in a high % that you can exit quickly
def check_price_anomaly(df):
    div_lookahead = 5
    spy_dividend = pd.read_csv('data/spy_dates_dividends.csv')
    spy_dividend['ex_date'] = pd.to_datetime(spy_dividend['ex_date'], format='%m/%d/%Y')
    index_day = date_to_index_day(spy_dividend['ex_date'], spy_data.data)
    spy_dividend['index_day'] = index_day['index_day']
    spy_dividend = spy_dividend.dropna()
    future_data = select_df_ranges_from_dates(spy_data.price_and_volume(), spy_dividend['index_day'], div_lookahead)
    split_indecies = future_data['index_day']

    # The mean of a days should be heavily negative (as they are after ex div date)
    # The mean of b days sohuld be slightly positive (normal market behaviour)
    a, b = spy_data.split_on_days(split_indecies)
    a = a.to_daily().sort_index()
    b = b.to_daily().sort_index()
    # a = a['close'] - a['open']
    # b = b['close'] - b['open']

    # Sanity check
    assert len(a) == div_lookahead * len(spy_dividend)

    ex_div_change = []
    # for i in range(len(spy_dividend))

    stacked_a = a.values.reshape((div_lookahead), len(spy_dividend))
    stacked_a = np.array([np.cumsum(x) for x in stacked_a])
    lookahead_mean = np.mean(stacked_a, axis=1)

    fig, ax = plt.subplots()
    ax.set_title('Change in price vs days since ex dividend')
    ax.set_xlabel('Days since ex dividend')
    ax.set_ylabel('Change in price ($)')
    ax.plot(lookahead_mean)
    plt.show(block=True)

    change = df.get_daily_price_change()
    print(f'mean: {np.mean(change)}')

# auto-arima for prediction
# https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

# ray start --head --port=6379
ray.init()

csv_files = [f'data/{x}' for x in os.listdir('data') if x.startswith('spy_stock_minute')]
spy_data = StockData(load_raw_data_from_multiple_csvs(csv_files))

check_price_anomaly(spy_data)

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