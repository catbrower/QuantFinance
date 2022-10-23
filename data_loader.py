from cmath import isnan
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import CCIIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from math_util import *

DATA_TO_DECIMAL = 4

default_indicators = {'rsi_period': 30, 'vwap_period': 46, 'atr_period': 39, 'macd_short': 21, 'macd_long': 4, 'macd_signal': 21, 'bb_period': 4, 'std': 33}

# 1.0
# {'rsi_period': 42, 'vwap_period': 54, 'atr_period': 50, 'macd_short': 43, 'macd_long': 53, 'macd_signal': 9, 'bb_period': 4, 'std': 16}

# Ue when fine tuning a model
def train_generator_fixed(data, lookback, X, Y, is_buy_indicator=True, valid_days=None):
    num_days = max(data['index_day'])
    # longest_indicator = max(indicators[x] for x in indicators)
    start_index = 390 - len(data[data['index_day'] == 0].dropna())

    while True:
        for index_day in range(num_days):
            #Valid days is a list that constrains what is a valid traing day
            if valid_days is not None and not index_day in valid_days:
                continue

            selection = data[data['index_day'] == index_day]
            if len(selection) != 390:
                print(f'warning, data for day %d is malformed' % index_day)
                continue

            # start_index = longest_indicator + lookback
            for index_minute in range(start_index, 390 - lookback):
                train_x = selection.iloc[index_minute:index_minute + lookback][X].values.tolist()
                if is_buy_indicator:
                    train_y = float(selection.iloc[index_minute + lookback]['binary_reward_buy'])
                else:
                    train_y = float(selection.iloc[index_minute + lookback]['binary_reward_sell'])
                yield np.array([train_x]), np.array([train_y])

# use for fast training at the cost of accuracy
# TODO google ways to improve accuracy while using batching
def batch_train_generator_fixed(data, lookback, X, Y, batch_size, is_buy_indicator=True, valid_days=None, num_days=None):
    generator = train_generator_fixed(
                data,
                lookback,
                X, Y,
                is_buy_indicator=is_buy_indicator,
                valid_days=valid_days)
    while True:
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            X, Y = generator.__next__()
            x_batch.append(X[0])
            y_batch.append(Y)
        yield np.array(x_batch), np.array(y_batch)

# prune the reward metric to remove noise
def prune_reward(data, prune_length=5):
    pruned = []
    index_start = 0
    index_end = 0
    for i in range(len(data)):
        if data[i] == 1:
            index_end += 1
        else:
            if index_end - index_start < prune_length:
                pruned.append([0] * (index_end - index_start + 1))
            else:
                pruned.append(data[index_start:index_end + 1])
            index_start = i
            index_end = i

    if index_start != index_end:
        pruned.append(data[index_start:index_end + 1])

    return [y for x in pruned for y in x]

def to_pct(values):
    return np.cumsum(values.rolling(2).apply(lambda x: (x.iloc[1] - x.iloc[0]) / x.iloc[0] * 100))

# Calculate function over dataframe by day
# return pd.Series
def calculate_indicators(df, indicators):
    reward_period = 30
    reward_threshold = 0.005

    # df = df.reset_index()
    # result = pd.Series(index=df['time'])
    # result.index = dataframe.index
    df['index_day'] = pd.Series(dtype=int)
    df['mean'] = df[['open', 'high', 'low', 'close']].mean(axis=1)

    df['open'] = np.log(df['open'])
    df['high'] = np.log(df['high'])
    df['low'] = np.log(df['low'])
    df['close'] = np.log(df['close'])
    end_date = max(df.index)
    date = min(df.index)
    next_date = date + timedelta(days = 1)
    day_number = 0

    prev_date = None
    while date < end_date:
        # data_selection
        # Calculate daily % value change
        selection = df[(df.index >= date) & (df.index < next_date)]

        # Skip weekends + holidays
        if len(selection) == 390:
            if prev_date is None:
                df.loc[selection.index, 'raw_volume'] = selection['volume'].values
                prev_date = date
                continue

            prev_selection = df[(df.index >= prev_date) & (df.index < prev_date + timedelta(days=1))]

            # # open = selection['open']
            raw_open = selection['open']
            raw_high = selection['high']
            raw_low = selection['low']
            raw_close = selection['close']
            raw_volume = selection['volume']
            raw_mean = selection[['open', 'high', 'low', 'close']].mean(axis=1)
            high   = to_pct(selection['high']).fillna(0)
            low    = to_pct(selection['low']).fillna(0)
            close  = to_pct(selection['close']).fillna(0)
            mean   = to_pct(raw_mean)
            volume = (raw_volume - prev_selection['raw_volume'][0:-1].mean()) / (prev_selection['raw_volume'][0:-1].std() * 10)
            
            # Determine Spread
            mean_spread = selection[['askopen', 'askhigh', 'asklow', 'askclose']].mean(axis=1) - selection[['bidopen', 'bidhigh', 'bidlow', 'bidclose']].mean(axis=1)
            prev_mean_spread = prev_selection[['askopen', 'askhigh', 'asklow', 'askclose']].mean(axis=1) - prev_selection[['bidopen', 'bidhigh', 'bidlow', 'bidclose']].mean(axis=1)
            mean_spread = mean_spread - prev_mean_spread.mean()
            mean_spread = mean_spread / (prev_mean_spread.std() * 10)

            # mean   = round(selection['mean'], DATA_TO_DECIMAL)
            # fd_mean = fractional_difference(mean, 0.5, 0.05)
            # returns = close.rolling(2).apply(lambda x: (x.iloc[1] - x.iloc[0]) / x.iloc[0] * 100).fillna(0)
            std = close.rolling(indicators['std'].value).apply(lambda x: np.std(x))
            # pct_change = np.cumsum(returns)

            # Calculate Indicators
            # day_index = pd.Series(data = [day_number] * len(close), index = selection.index, name='index_day')
            rsi = RSIIndicator(close, window=indicators['rsi_period'].value).rsi() / 100
            vwap = VolumeWeightedAveragePrice(high, low, close, raw_volume, window=indicators['vwap_period'].value).volume_weighted_average_price()
            # cci = CCIIndicator(high, low, close, window=cci_period).cci()
            macd = MACD(close, indicators['macd_long'].value, indicators['macd_short'].value, indicators['macd_signal'].value).macd()
            atr = AverageTrueRange(high, low, close, window=indicators['atr_period'].value).average_true_range()
            bb = BollingerBands(close, indicators['bb_period'].value)

            # Calculate Reward
            # r_series = pd.Series(pct_change.iloc[::-1].values)
            dwt = dwt_denoise(close, level=4)
            r_series = pd.Series(dwt[::-1])
            # r_series = pd.Series(r_series.ewm(com=reward_period).mean().iloc[::-1].values)
            r_series = pd.Series(r_series.rolling(reward_period).mean().iloc[::-1].values - reward_threshold)
            
            reward_buy = (r_series - close.values).fillna(0).values
            binary_reward_buy = [1 if x > 0 else 0 for x in reward_buy]
            # binary_reward_buy = prune_reward(binary_reward_buy, prune_length=5)

            reward_sell = (close.values - r_series).fillna(0).values
            binary_reward_sell = [1 if x > 0 else 0 for x in reward_sell]
            # binary_reward_sell = prune_reward(binary_reward_sell, prune_length=5)

            # Set new values
            df.loc[selection.index, 'raw_open'] = raw_open
            df.loc[selection.index, 'raw_high'] = raw_high
            df.loc[selection.index, 'raw_low'] = raw_low
            df.loc[selection.index, 'raw_close'] = raw_close
            df.loc[selection.index, 'raw_mean']  = raw_mean
            df.loc[selection.index, 'raw_volume'] = raw_volume
            df.loc[selection.index, 'high'] = high
            df.loc[selection.index, 'low'] = low
            df.loc[selection.index, 'close'] = close
            df.loc[selection.index, 'mean'] = mean
            df.loc[selection.index, 'volume'] = volume
            df.loc[selection.index, 'index_day'] = int(day_number)
            df.loc[selection.index, 'index_minute'] = np.arange(len(selection))
            df.loc[selection.index, 'mean_spread'] = mean_spread
            # df.loc[selection.index, 'returns'] = returns
            # df.loc[pct_change.index, 'pct_mean'] = pct_change
            df.loc[selection.index, 'std'] = std
            df.loc[selection.index, 'rsi'] = rsi
            df.loc[selection.index, 'vwap'] = vwap
            # df.loc[selection.index, 'cci'] = cci
            df.loc[selection.index, 'macd'] = macd
            df.loc[selection.index, 'atr'] = atr
            df.loc[selection.index, 'bb_hband'] = bb.bollinger_hband_indicator()
            df.loc[selection.index, 'bb_lband'] = bb.bollinger_lband_indicator()
            df.loc[selection.index, 'dwt'] = dwt
            df.loc[selection.index, 'reward_buy'] = reward_buy
            df.loc[selection.index, 'binary_reward_buy'] = binary_reward_buy
            df.loc[selection.index, 'reward_sell'] = reward_sell
            df.loc[selection.index, 'binary_reward_sell'] = binary_reward_sell

            prev_date = date
            day_number += 1

        date += timedelta(days = 1)
        next_date += timedelta(days = 1)

    # Filter out any rows where index_day == nan
    print('Data loaded')
    df = df[df['index_day'] < float('inf')]
    df = df.astype({'index_day': int})
    return df

def load_data(use_cached=False, indicators=default_indicators):
    if use_cached:
        return pd.read_csv('data_cache.csv')
        
    data = pd.read_csv('data/spy_stock_minute_2019-01-01_2020-01-01.csv')
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data = data.set_index('time')

    data = calculate_indicators(data, indicators)

    return data