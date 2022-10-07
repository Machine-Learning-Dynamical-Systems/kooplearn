#Misc
import sys
sys.path.append("../../")
#Torch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import datasets
from torchvision.transforms import ToTensor
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

image_dim =  28 #MNIST specific

def build_sequential_data(targets=_train_data.targets, num_classes=10):
    assert num_classes <= 10
    keys = np.arange(num_classes)
    vals = [[] for _ in range(num_classes)]
    _data = dict(zip(keys, vals))
    for _idx in range(len(targets)):
        sample_idx = targets[_idx].item()
        if sample_idx < num_classes:  
            _data[sample_idx].append(_idx)
    #_data is a dict with num_classes keys. _data[key] is a list of indexes of images representing the digit ``key''.
    min = np.Inf
    for k in _data.keys():
        array_len = len(_data[k])
        if array_len  < min:
            min = array_len
    cols = []
    for k in _data.keys():
        cols.append(_data[k][:min])
    return np.column_stack(cols).flatten()

data = _train_data.data.float().numpy()
targets = _train_data.targets.int().numpy()