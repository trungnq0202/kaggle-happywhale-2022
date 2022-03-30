import numpy as np
import torch
import torchvision
import os



def vprint(obj, vb):
    if vb:
        print(obj)
    return