import time
import ray
import numpy as np

from data_loader import *
from Models import FinanceModel
from Database import Database
import Models.HyperParameter as HyperParameter

class Tuner():
    def __init__(self, initial_indicators, columns_y):
        self.columns_x = [x for x in initial_indicators]
        self.columns_y = columns_y
        self.indicators = initial_indicators
        self.num_threads = 1
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
        def worker(indicators, Y):
            data = load_data(indicators=indicators)

            try:
                model = FinanceModel.FinanceModel(indicators, Y, data)
                # model.set_indicators(indicators)

                # return None
                # TODO epochs should be in here
                history = model.fit()
                # return (indicators, history)
                return None
            except Exception as ex:
                print(ex)
                return (indicators, None)
                
        jobs = []
        for _ in range(self.num_threads):
            jobs.append(worker.remote(self.indicators, 'binary_reward_buy'))

        return ray.get(jobs)

    def randomize_models(self, models):
        def worker(Y, model):
            weights = [np.array(x) for x in model['model']]
            indicators = {}
            for indicator in model['indicators']:
                indicators[indicator] = HyperParameter.HyperParameter(2, 60, 1, model['indicators'][indicator])

            data = load_data(indicators=indicators)

            scale = 1
            epochs = 10

            for i in range(epochs):
                model = FinanceModel.FinanceModel.from_weights(weights, indicators, Y, data)
                model.randomize_weights(scale * np.exp(i - epochs))
                results = model.predict()

                # sanity check
                # the number of predictions should equal the # of none NA data points in each day * # days
                # if this isn't true something is probably wrong
                values_in_day = len(model.dataset[model.dataset['index_day'] == 0].dropna())
                assert len(results) == max(model.dataset['index_day']) * values_in_day

                # for each day, calculate performance
                for i in range(max(model.dataset['index_day'])):
                    day_values = model.dataset[model.dataset['index_day'] == i]
                    day_predictions = results[i * values_in_day, (i + 1) * values_in_day]
                    day_predictions_binary = np.sign(day_predictions - 0.5)
                    day_predictions_binary[np.argwhere(day_predictions_binary < 0)] = 0
                    prediction_diff = day_predictions_binary - model.dataset['binary_reward_buy']
                    FP = np.argwhere(prediction_diff == 1)
                    FN = np.argwhere(prediction_diff == -1)
                    TP = [day_predictions[x] == 1 for x in np.argwhere(prediction_diff == 0)]
                    TN = [day_predictions[x] == 0 for x in np.argwhere(prediction_diff == 0)]

                    recall = TP / (TP + FN)
                    precision = TP / (TP + FP)
                    accuracy = (TP + TN) / len(day_predictions)

                # evaluate
                # predictions = model.predict()
                # save to db with randomized = true
            return 0

        @ray.remote
        def ray_worker(Y, model):
            return worker(Y, model)

        worker('binary_reward_buy', models[0])

        jobs = []
        for i in range(self.num_threads):
            jobs.append(ray_worker.remote('binary_reward_buy', models[i]))
        ray.get(jobs)
        pass

    def search(self, iterations):
        models = self.db.client.finance.models
        for _ in range(iterations):
            self.db.calculate_stats()
            # self.tune_indicators()

            while self.db.model_mean is None or self.db.model_std is None:
                time.sleep(0.1)
            best_models = self.db.get_random_best_models(self.num_threads)
            self.randomize_models(best_models)

            