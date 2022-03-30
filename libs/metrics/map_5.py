import torch
import numpy as np

class MAP5():
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        map_subset = []
        for (prediction, label) in zip(output, target):
            map_per_img = 0.0
            try:
                map_per_img = 1 / (prediction[:5].index(label) + 1)
            except:
                map_per_img = 0.0

            map_subset.append(map_per_img)
        return map_subset

    def update(self, value):
        self.map_all_set = self.map_all_set + value
        
    def reset(self):
        self.map_all_set = []

    def value(self):
        return np.mean(self.map_all_set)

    def summary(self):
        print(f'MAP@5 of all set: {self.value()}')
