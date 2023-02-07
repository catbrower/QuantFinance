import numpy as np

class HyperParameter:
    def __init__(self, value_min, value_max, value_step, value=None):
        self.value_min = value_min
        self.value_max = value_max
        self.value_step = value_step
        self.value = value_min if value is None else value
        self.prev_values = []

    def get_random_value(self):
        result = np.random.uniform(self.value_min, self.value_max)
        return self.value_step * int(result / self.value_step)

    def next_value(self):
        if self.value is None:
            self.value = self.get_random_value()
        else:
            if np.random.rand() < 0.5:
                self.value -= self.value_step
            else:
                self.value += self.value_step
        return self.value