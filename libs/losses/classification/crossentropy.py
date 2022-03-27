import torch
import torch.nn as nn

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, **kwargs):
        if weight is not None:
            weight = torch.FloatTensor(weight)
        super().__init__(weight, **kwargs)