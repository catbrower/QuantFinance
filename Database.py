from pymongo import MongoClient

class Database():
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Database, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.client = MongoClient('localhost', 27017)

    def get_instance(self):
        return self.client

    def get_all_model_stats(self):
        return self.client['finance']['models'].find({}, {'model': 0})