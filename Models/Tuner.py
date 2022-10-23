import ray
import numpy as np

from data_loader import *
from Models import FinanceModel
from Database import Database
import Models.HyperParameter

class Tuner():
    def __init__(self, columns_x, columns_y, initial_indicators):
        self.columns_x = columns_x
        self.columns_y = columns_y
        self.indicators = initial_indicators
        self.num_threads = 14
        self.db = Database()

    # TODO implement scale
    def random_indicators(self, scale=1):
        result = {}
        for key in self.indicators:
            indicator = self.indicators[key]
            result[key] = int(np.random.rand() * indicator.max) + indicator.min

        return result

    def tune_indicators(self):
        @ray.remote
        def worker(X, Y, indicators):
            data = load_data(indicators=indicators)

            try:
                model = FinanceModel.FinanceModel(data)
                model.set_indicators(indicators)

                history = model.fit()
                return (indicators, history)
            except Exception as ex:
                print(ex)
                return (indicators, None)
                
        jobs = []
        for _ in range(self.num_threads):
            jobs.append(worker.remote(None, 'binary_reward_buy', self.indicators))

        return ray.get(jobs)

    def search(self, iterations):
        models = self.db.client.finance.models
        for _ in range(iterations):
            self.db.calculate_stats()
            # self.tune_indicators()
            best_models = self.db.get_random_best_models(self.num_threads)
            print()
            # select top models
            # randomize on cylcical schedule
            # 

            