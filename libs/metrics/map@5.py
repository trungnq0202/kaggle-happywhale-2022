import torch
import numpy as np

class MAP5():
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        for (prediction, label) in zip(output, target):
            map_per_img = 0.0
            try:
                map_per_img = 1 / (prediction[:5].index(label) + 1)
            except:
                map_per_img = 0.0

            self.map_all_set.append(map_per_img)

    def update(self, value):
        raise NotImplementedError

    def reset(self):
        self.map_all_set = []

    def value(self):
        return np.mean(self.map_all_set)

    def summary(self):
        print(f'MAP@5 of all set: {self.value()}')
