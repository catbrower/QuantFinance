import re

import ray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, boxcox
from ta.volume import VolumeWeightedAveragePrice

from data_loader import set_time_colume_to_date, calculate_day_and_minute_index
from Util import dwt_denoise

PROCESSING_CHUNKS = 20
TIME_COLUMNS = ['index_day', 'index_minute']

def load_raw_data_from_multiple_csvs(files):
    dataframes = []
    for csv_file in files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)
    data = set_time_colume_to_date(data)
    data = calculate_day_and_minute_index(data)
    return data

@ray.remote
def apply_function_by_day(df, func):
    return apply_function_by_day_sync(df, func)

def apply_function_by_day_sync(df, func):
    results = []
    for day_data in split_daily(df):
        results.append(func(day_data))
    return pd.concat(results)

# have to use a custom function in order to ensure that data for some day isnt' split into different chunks
# will return multiple chunks containing multple daily data. Individual days will not be split across chunks
def chunk_data_by_day(df, num_chunks):
    if 'index_day' not in df.columns:
        raise Exception('cannot chunk data by day if index_day not ini columns')
    
    # do it this way just in case index_day isn't contiguous for some reason
    unique_days = pd.unique(df['index_day'])
    split_days = np.array_split(unique_days, num_chunks)

    return [df[df['index_day'].isin(split_days[i])] for i in range(len(split_days))]

# split a df into parts by index_day
# will return one df for each day
def split_daily(df):
    return [df[df['index_day'] == index_day] for index_day in pd.unique(df['index_day'])]

# Calcualte daily stats for day
def calculate_statistics_for_day(daily_data):
        # filter out time columns here
        index = pd.Series(daily_data['index_day'].iloc[0])
        cols = [c for c in daily_data.columns if c not in TIME_COLUMNS]
        return pd.concat([
            pd.DataFrame({f'std_{c}': np.std(daily_data[c].dropna()) for c in cols}, index=index),
            pd.DataFrame({f'mean_{c}': np.mean(daily_data[c].dropna()) for c in cols}, index=index),
            pd.DataFrame({f'kurt_{c}': np.mean(kurtosis(daily_data[c].dropna())) for c in cols}, index=index),
            pd.DataFrame({f'skew_{c}': np.mean(skew(daily_data[c].dropna())) for c in cols}, index=index)
        ], axis=1)

class PriceTuple:
    def from_dataframe(df, zero_data=True):
        return PriceTuple(
            df['open'],
            df['high'],
            df['low'],
            df['close'],
            df['volume'],
            zero_data=zero_data
        )

    def __init__(self, O, H, L, C, V, zero_data=True):
        self.O = O
        self.H = H
        self.L = L
        self.C = C
        self.V = V
        self.M = (O + H + L + C) / 4

        # if zero_data:
        #     self.O -= self.O.iloc[0]
        #     self.H -= self.H.iloc[0]
        #     self.L -= self.L.iloc[0]
        #     self.C -= self.C.iloc[0]
        #     self.M -= self.M.iloc[0]

class Indicator:
    STD = 'std'
    VWAP = 'vwap'
    MAVG = 'mavg'

    def __init__(self, indicator_type, indicator_length):
        self.indicator_type = indicator_type.lower()
        self.indicator_length = indicator_length
        self.value = None
        self.normalized_value = None

        if(indicator_type.lower() == Indicator.VWAP):
            self.calculate = lambda price_tuple: VolumeWeightedAveragePrice(price_tuple.H, price_tuple.L, price_tuple.C, price_tuple.V, window=self.indicator_length).volume_weighted_average_price()
        elif(indicator_type.lower() == Indicator.STD):
            self.calculate = lambda price_tuple: price_tuple.M.rolling(self.indicator_length).std()
        elif(indicator_type.lower() == Indicator.MAVG):
            self.calculate = lambda price_tuple: price_tuple.M.rolling(self.indicator_length).mean()
        else:
            raise Exception(f"Unknown indicator type: {indicator_type}")

    def get_value(self, price_tuple):
        result = self.calculate(price_tuple)
        
        self.value = result
        return result
    
    def get_normalized_value(self, price_tuple, mean, std):
        if self.normalized_value is not None:
            return self.normalized_value
        
        result = self.get_value(price_tuple)
        return (result - mean) / std

    def column_name(self):
        return f'{self.indicator_type}_{self.indicator_length}'
    
    def length(self):
        return self.indicator_length
    
    def type(self):
        return self.indicator_type

class IndicatorSpec:
    def __init__(self, spec=[], normalized=True):
        self.spec = []
        self.normalized = normalized

        norm_data = {}
        for indicator in spec:
            if not self.hasIndicatorWithWindow(indicator[0], indicator[1]):
                # Add STD and mean values for normalization
                if(self.normalized):
                    if indicator[1] not in norm_data:
                        norm_data[indicator[1]] = (
                            Indicator(Indicator.MAVG, indicator[1]),
                            Indicator(Indicator.STD, indicator[1])
                        )
                    self.addIndicator(indicator[0], indicator[1], norm_values = norm_data[indicator[1]])
                else:
                    self.addIndicator(indicator[0], indicator[1])

    def isNormalized(self):
        return self.normalized
        
    def hasIndicatorWithWindow(self, indicator_type, indicator_length):
        for indicator in self.spec:
            if indicator.indicator_type == indicator_type and indicator_length == indicator_length:
                return True
            return False

    def addIndicator(self, indicator, length, norm_values=None):
        self.spec.append(Indicator(indicator, length))

    def vwap(self, length):
        self.addIndicator(Indicator.VWAP, length)

    def std(self, length):
        self.addIndicator(Indicator.STD, length)

    def mavg(self, length):
        self.addIndicator(Indicator.MAVG, length)

    def toList(self):
        return self.spec

class DataSet:
    def __init__(self, data):
        self.time_columns = ['index_minute', 'index_day']
        self.ohlcmv_columns = ['open', 'high', 'low', 'close', 'mean', 'volume']
        self.training_set_lookback = 60

        # TODO this isn't a good pattern
        if data is not None:
            # Validation
            if not set(self.time_columns).issubset(data.columns):
                raise Exception('Cannot create a dataset without the necessary time coluns')
            self.data = data

    def set_data(self, data):
        # TODO bad pattern
        # this doesn't apply to daily statistics anyway because it has not minutes
        # if not set(self.time_columns).issubset(data.columns):
        #     raise Exception('Cannot create a dataframe without the necessary time')
        self.data = data

    def columns(self):
        return self.data.columns
    
    def length(self):
        return len(self.data)
    
    def get_day_coverage(self):
        return pd.unique(self.data['index_day'])
    
    def get_data_for_day(self, index_day):
        return self.data[self.data['index_day'] == index_day]

    def get_data_in_day_range(self, index_day_start, index_day_end):
        return self.data[
            (self.data['index_day'] >= index_day_start)
            & (self.data['index_day'] <= index_day_end)
        ]

    def to_training_set(self):
        result = []

        for day in pd.unique(self.data['index_day']):
            day_data = self.data[self.data['index_day'] == day].dropna(axis=1)
            day_data = day_data.reset_index()
            day_data = day_data.drop(['time'], axis=1)
            for i in range(min(day_data['index_minute']), max(day_data['index_minute']) - self.training_set_lookback):
                # Might want to drop time columns here
                result.append((day_data['index_day'] >= i) & (day_data['index_day'] < i + self.training_set_lookback))

        return result
    
    def get_non_time_columns(self):
        return [x for x in self.data.columns if x not in TIME_COLUMNS]

    def plot_intraday_value(self, column):
        plt.plot(self.data[column].index, self.data[column])

    def plot_interday_value(self, column, index_day):
        plt.plot(self.get_data_for_day(index_day)[column])

    # TODO remove
    @ray.remote
    def call_daily_func_on_chunk(self, chunk):
        results = []
        for daily_df in split_daily(chunk):
            results.append(self.process_day(daily_df))

        return pd.concat(results)
    
    #TODO remove
    def process_day(self, daily_df):
        pass

class DailyStatistics(DataSet):
    def __init__(self, indicator_data):
        super().__init__(None)
        results = [apply_function_by_day.remote(c, calculate_statistics_for_day) for c in chunk_data_by_day(indicator_data, PROCESSING_CHUNKS)]
        results = ray.get(results)
        results = pd.concat(results).sort_index()
        self.set_data(results)

# indicator data should only be created from stock_data, maybe add some checks in case I forget this
class IndicatorData(DataSet):
    def __init__(self, stock_data, indicator_spec):
        super().__init__(None)
        self.stock_data = stock_data
        self.indicator_spec = indicator_spec

        # TODO this is throwing a lot of warnings. Some indexinf method that I'm using is deprecated
        # TODO this has to be revised. I need to use a method thats properly decorated with @ray.remote not an anon func
        # TODO this may not be the best way to do this since the chunk func is on the class
        # it might end up serializing th whole obj which we don't want
        # soln would be to take it out of the obj and pass in the daily_func to be evaluated
        chunks = chunk_data_by_day(stock_data, PROCESSING_CHUNKS)
        results = [apply_function_by_day.remote(c, self.process_day) for c in chunks]
        results = pd.concat(ray.get(results))
        results[self.time_columns] = stock_data[self.time_columns]
        self.set_data(results)
        
        # self.daily_statistics = daily_statistics
        # self.required_cols = ['index_minute', 'index_day', 'std', 'mean']

    # put daily stats calculations in here
    # TODO daily statistics have been move to a seperate class, check that this still works
    def process_day(self, data_day):
        if len(pd.unique(data_day['index_day'])) > 1:
            raise Exception('compute_indicators_for_day has been passed data with multiple days in it')
        
        index_day = data_day['index_day'].iloc[0]
        price_tuple = PriceTuple.from_dataframe(data_day[self.ohlcmv_columns])
        # log = np.log(price_tuple.M)
        # pct = log / log[0]
        # pct_returns = pct - pct.shift(1)
        # dwt = dwt_denoise(pct, level=4)
        result = {
            'index_day': index_day,
            'index_minute': np.arange(390)
        }

        for indicator in self.indicator_spec.toList():
            result[indicator.column_name()] = indicator.get_value(price_tuple).values

        indicators = pd.DataFrame(result, index = data_day.index)
        return indicators

    def indicator(self, type, day=None):
        cols = []
        for col in self.data.columns:
            if re.search(rf'{type}', col, re.IGNORECASE):
                cols.append(col)
        
        if day:
            result = self.data[self.data['index_day'] == day]

        return IndicatorData(indicator_data=result, daily_statistics=self.daily_statistics)

    def compute_daily_statistics(self):
        result = DailyStatistics(self.data)
        return result

class NormalizedIndicatorData(IndicatorData):
    def __init__(self, indicator_data, daily_stats):
        self.indicator_data = indicator_data
        self.daily_stats = daily_stats

        # Let's just start with a 5 day look back and try to mornalize based on that
        cols = indicator_data.get_non_time_columns()
        lookback = 5
        

        print()

class StockData(DataSet):
    def __init__(self, data):
        super().__init__(data)
        self.initial_columns = data.columns
        data = self.calculate_mean(data)
        self.data = data
        self.num_days = max(data['index_day'])

        # Just here for posterity
        self.default_indicator_spec = IndicatorSpec([
            ('MAVG', 60),
            ('MAVG', 30)
        ])

    def calculate_mean(self, data):
        data['mean'] = data[['open', 'high', 'low', 'close']].mean()
        return data

    def get_olhcv_for_day_index(self, index_day):
        return PriceTuple(
            self.data[self.data['index_day'] == index_day]['open'],
            self.data[self.data['index_day'] == index_day]['high'],
            self.data[self.data['index_day'] == index_day]['low'],
            self.data[self.data['index_day'] == index_day]['close'],
            self.data[self.data['index_day'] == index_day]['volume']
        )
    
    @ray.remote
    def compute_indicators_for_chunk(self, day_start, day_end, indicator_spec):
        results_indicator = []
        results_stats = []
        for i in range(day_start, day_end):
            data_day = self.compute_indicators_for_day(i, indicator_spec)
            results_indicator.append(data_day[0])
            results_stats.append(data_day[1])
        return (
            pd.concat(results_indicator),
            pd.concat(results_stats)
        )

    def normalize_indicators_for_day(data, daily_statistics, index_day):
        norm_lookback = 5
        for column in data.columns:
            if column == 'index_day' or column == 'index_minute':
                pass

            mean = np.mean(daily_statistics[(daily_statistics['index_day'] < index_day) & (daily_statistics['index_day'] >= index_day - norm_lookback)]['mean'])
            std = np.mean(daily_statistics[(daily_statistics['index_day'] < index_day) & (daily_statistics['index_day'] >= index_day - norm_lookback)]['std'])
            data[column] = (data[column] - mean) / std

        return data
    
    # see TODO below, this functionality is so simple I can def put it into some kind of
    # common method
    @ray.remote
    def normalize_indicators_chunk(self, indicator_data, daily_statistics, day_start, day_end):
        results = []
        for i in range(day_start, day_end):
            data_day = self.normalize_indicators_for_day(indicator_data, daily_statistics, i)

        return pd.concat(results)

    # TODO: I can probably extract the chunking functionality into a common funtion
    #  it could work by passing in some dataframe and a num_chunks variable then split it up like that
    #  could also create a function for splitting based on some column by unique values, that would be just
    #  a general purpose version of splittling on index day.
    #  An additional benefit to doing the data splitting method is that it would prevent lookahead, i.e the
    #  daily functions would be passed only the data that they could actually see
    # Time periods is a list of ints that specify which lookback window to use for computing indicators
    def compute_indicators(self, indicator_spec=None):
        if(indicator_spec is None):
            indicator_spec = self.default_indicator_spec

        indicator_data = IndicatorData(self.data, indicator_spec)
        return indicator_data
