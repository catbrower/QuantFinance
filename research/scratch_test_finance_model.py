import os

import ray

from Database import Database
from data_loader import *
from Models import FinanceModel

# Disable tensorflow logs because they're excessive and annoying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def worker(X, Y):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    indicators = generate_indicators()
    data = load_data(indicators=indicators)

    try:
        model = FinanceModel.FinanceModel(data)
        model.set_indicators(indicators)

        history = model.fit()
        return (indicators, history)
    except Exception as ex:
        print(ex)
        return (indicators, None)

@ray.remote
def ray_worker(X, Y):
    return worker(X, Y)

@ray.remote
def ray_run_model(X, Y):
    result = worker(X, Y)
    return result

def generate_indicators():
    result = {}
    for key in default_indicators:
        result[key] = int(np.random.rand() * 59) + 2

    return result



# num_threads = 14
# for _ in range(100):
#     try:
#         jobs = []
#         for _ in range(num_threads):
#             jobs.append(ray_worker.remote(None, 'binary_reward_buy'))

#         results = ray.get(jobs)

#         na_results = [x for x in results if x[1] is not None]

#         print(na_results)
#         if len(na_results) == 0:
#             continue
        
#         best_index = np.argmax([max(x[1]['precision']) for x in na_results])
#         best_indicators = na_results[best_index][0]
#         best_value = max(na_results[best_index][1]['precision'])

#         print(best_value)
#         print(best_indicators)
#         print('-' * 50)
#     except:
#         pass