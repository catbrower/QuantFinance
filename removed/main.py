import datetime

import ray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Day

DATE_FORMAT = '%Y-%m-%d'
NUM_THREADS = 12

data = pd.read_csv('data/spy_stock_minute_2019-01-01_2020-01-01.csv')
data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
# data = data.set_index('time')

# For each day
day = datetime.datetime.strptime('2018-01-01', DATE_FORMAT)
date_end = datetime.datetime.strptime('2019-01-01', DATE_FORMAT)

@ray.remote
def batch_trade(days):
    for day in days:
        day.trade()

    return days

days = []
while day < date_end:
    # calc mean for testing
    data_day = data.loc[(data['time'] > day) & (data['time'] < (day + datetime.timedelta(days = 1)))]

    if not len(data_day) == 0:
        days.append(Day.Day(data_day))
        # day_max = max(data_day['close'])
        # day_min = min(data_day['close'])
        # day_open = float(data_day[data_day['time'] == min(data_day['time'])]['close'])

        # if day_open == day_min or day_open == day_max:
        #     num_conditions_met += 1

        # num_days += 1
    day = day + datetime.timedelta(days = 1)

# Partition data
partition_size = np.ceil(len(days) / NUM_THREADS)
# partitions = [batch_trade(days[int(i * partition_size): int((i + 1) * partition_size)]) for i in range(NUM_THREADS)]
work = [batch_trade.remote(days[int(i * partition_size): int((i + 1) * partition_size)]) for i in range(NUM_THREADS)]

results = ray.get(work)
results = [y for x in results for y in x]

plt.hist([day.profit for day in results], bins=50)
plt.show(block=True)