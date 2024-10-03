import torch
import numpy as np

def mae(x, y):
    
    if isinstance(x, torch.Tensor):
        return torch.abs(x - y).sum(0)
    else:
        return np.abs(x - y).sum(0)

def mse(x, y):

    if isinstance(x, torch.Tensor):
        return ((x-y)*(x-y)).sum(0)
    else:
        return ((x-y)*(x-y)).sum(0)


