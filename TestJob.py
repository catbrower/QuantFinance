import time

import numpy as np

from Jobs.EndlessJob import EndlessJob

class TestJob(EndlessJob):
    def __init__(self):
        self.name = "TestJob"
    # def __iter__(self):
    #     super.__iter__(self, 'test_job')

    def next_task(self):
         return np.random.random() + 1

    def do_task(self, args):
        time.sleep(args)

    def post_task(self):
        pass