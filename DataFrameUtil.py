import ray
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, boxcox

from data_loader import set_time_colume_to_date, calculate_day_and_minute_index
from Util import dwt_denoise

TIME_COLUMNS = ['index_day', 'index_minute']

@ray.remote
def apply_function_by_day(df, func):
    return apply_function_by_day_sync(df, func)

def load_raw_data_from_multiple_csvs(files):
    dataframes = []
    for csv_file in files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)
    data = set_time_colume_to_date(data)
    data = calculate_day_and_minute_index(data)
    return data

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
def calculate_statistics(daily_data):
    # filter out time columns here
    index = pd.Series(daily_data['index_day'].iloc[0])
    cols = [c for c in daily_data.columns if c not in TIME_COLUMNS]
    return pd.concat([
        pd.DataFrame({f'std_{c}': np.std(daily_data[c].dropna()) for c in cols}, index=index),
        pd.DataFrame({f'mean_{c}': np.mean(daily_data[c].dropna()) for c in cols}, index=index),
        pd.DataFrame({f'kurt_{c}': np.mean(kurtosis(daily_data[c].dropna())) for c in cols}, index=index),
        pd.DataFrame({f'skew_{c}': np.mean(skew(daily_data[c].dropna())) for c in cols}, index=index)
    ], axis=1)