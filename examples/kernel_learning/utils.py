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

class feature_map_cnn(torch.nn.Module):
    def __init__(self, out_features=32, clamp_last_layer = True, min=-0.5, max = 0.5):
        super(feature_map_cnn, self).__init__()
        self.num_features = out_features
        self.clamp_last_layer = clamp_last_layer
        self.min = min
        self.max = max
        self.conv1 = torch.nn.Sequential(         
            torch.nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            torch.nn.ReLU(),                      
            torch.nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = torch.nn.Sequential(         
            torch.nn.Conv2d(16, 32, 5, 1, 2),     
            torch.nn.ReLU(),                      
            torch.nn.MaxPool2d(2),                
        )
        # fully connected layer, output dim num_features
        self.out = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, self.num_features)
        )   
    def forward(self, x):
        x = torch.reshape(x, (-1, 1, image_dim, image_dim)) # Expect flattened data of shape [n_samples, image_dim**2]

        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        
        output = self.out(x)*(self.num_features**-0.5)
        if self.clamp_last_layer:
            return torch.clamp(output, min = self.min, max = self.max)    # Bound inputs
        else:
            return output

data = _train_data.data.float().numpy()
targets = _train_data.targets.int().numpy()