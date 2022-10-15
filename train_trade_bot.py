from tensorflow import keras
import matplotlib.pyplot as plt

from data_loader import *

buy_model = keras.models.load_model('buy_indicator.model')
sell_model = keras.models.load_model('sell_indicator.model')

data = load_data()

# Just pick some random day
index_day = 30
longest_indicator = 48
columns = [
    'pct_change',
    'rsi',
    'vwap',
    'macd',
    'atr'
]

def predict_generator():
    selection = data[data['index_day'] == index_day]
    if len(selection) != 390:
        print(f'warning, data for day %d is malformed' % index_day)
        return

    for index_minute in range(longest_indicator, 390):
        train_x = selection.iloc[longest_indicator:index_minute + 1][columns].values.tolist()
        train_y = float(selection.iloc[index_minute]['binary_reward'])
        yield np.array([train_x]), np.array([train_y])

buy_predicts = buy_model.predict_generator(predict_generator(), steps = 390 - longest_indicator)
buy_predicts = [1 if x[0] > 0.75 else 0 for x in buy_predicts]


plt.plot(data[data['index_day'] == index_day]['close'])
plt.show(block = True)