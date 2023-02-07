import numpy as np
import pickle

from numpy_fracdiff import fracdiff
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras

from data_loader import *
from math_util import *
from Database import Database

db = Database()
# stats = db.get_all_model_stats()
stats = db.get_instance()['finance']['models'].find()
# stats = np.array([x for x in stats])

# for stat in stats
precision = [x['metrics']['precision'] for x in stats]
mean = np.mean(precision)
std = np.std(precision)

top_precision = [x for x in precision if x > mean + std]

# plt.hist([x['metrics']['precision'] for x in stats], bins=20)
# plt.show(block=True)

print()