#Misc
import sys
sys.path.append("../../")
#Torch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
#Numpy-Matplotlib-tqdm
import numpy as np

_train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
_test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

def build_sequential_data(targets=_train_data.targets, num_classes=10):
    assert num_classes <= 10
    keys = np.arange(num_classes)
    vals = [[] for _ in range(num_classes)]
    _data = dict(zip(keys, vals))
    _inserted_indices = 0
    for _idx in range(len(targets)):
        sample_idx = targets[_idx].item()
        if sample_idx < num_classes:  
            _data[sample_idx].append(_idx)
            _inserted_indices += 1
    data = []
    _catch_exception = False
    while not _catch_exception:
        try:
            _target = _idx % num_classes   
            data.append(_data[_target][-1])
            _data[_target].pop()
            _idx += 1
        except Exception as e:
            break
    return np.array(data)

data = _train_data.data.float().numpy()