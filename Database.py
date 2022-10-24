import threading
from bson import ObjectId

import numpy as np
from pymongo import MongoClient

class Database():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Database, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.client = MongoClient('localhost', 27017)
        self.model_std = None
        self.model_mean = None
        self.stats_lock = threading.Lock()
        self.thread = None

    # locking immediatley is intentional here
    def calculate_stats(self):
        def _get_stats():
            self.stats_lock.acquire()
            results = list(self.client.finance.models.find({}, {'model': 0}))
            self.model_std = np.std([x['metrics']['precision'] for x in results])
            self.model_mean = np.mean([x['metrics']['precision'] for x in results])
            self.stats_lock.release()

        self.thread = threading.Thread(target=_get_stats)
        self.thread.start()

    def get_random_best_models(self, count):
        stats = self.get_stats()
        results = self.client.finance.models.find({'metrics.precision': {'$gt':stats['mean'] + stats['std']}, }, {'model': 0})
        ids = np.random.choice([x['_id'] for x in results], count)
        ids = [ObjectId(str(x)) for x in ids]
        results = self.get_instance().finance.models.find({'_id': {'$in': ids}})
        results = list(results)
        return results

    def get_stats(self):
        self.stats_lock.acquire()
        results = {
            'std': self.model_std,
            'mean': self.model_mean
        }
        self.stats_lock.release()
        return results

    def get_instance(self):
        return self.client

    def get_all_model_stats(self):
        return self.client['finance']['models'].find({}, {'model': 0})