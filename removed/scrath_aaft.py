from json import load
import numpy as np
import matplotlib.pyplot as plt

from data_loader import *
from math_util import *

index_day = 0
data = load_data()
selection = data[data['index_day'] == index_day]

aaft = aaft(selection['returns'])

plt.plot(selection['pct_close'])
plt.plot(selection.index, 
aaft)
plt.show(block = True)