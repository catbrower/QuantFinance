import matplotlib.pyplot as plt

from data_loader import *
from math_util import *
from tensorflow import keras
import matplotlib.pyplot as plt

data = load_data()
model = keras.models.load_model('buy_indicator.model')

selection = data[data['index_day'] == 0]
up = selection[selection['binary_reward'] == 1]['close']
plt.scatter(x=up.index, y=up)
plt.show(block=True)