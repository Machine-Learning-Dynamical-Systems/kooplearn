from functools import partial
import pandas as pd
import torch
from torch.utils.data import DataLoader
from kooplearn.models.feature_maps.dpnets import DPNet
from lightning.pytorch import Trainer
from kooplearn.nn.data import TrajToContextsDataset
from kooplearn.datasets.stochastic import LinearModel
from kooplearn.models.deepedmd import DeepEDMD
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from kooplearn.data import traj_to_contexts

# generating Datasets

seed = 42
rng = np.random.default_rng(seed)
# num_features = 10  # number of time series
train_val_split=0.5 #

# Deep Encoder model

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, flatten_input):
        super(MLPModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.flatten_input = flatten_input
        
        layers = [torch.nn.Linear(input_dim, hidden_dims[0]),
                 torch.nn.ReLU()]
        
        for i in range(len(hidden_dims)-1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(torch.nn.ReLU())
        
        layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        
        self.mlp = torch.nn.Sequential(*layers)
            
    def forward(self, X):
        if self.flatten_input:
            X = X.flatten()
        return self.mlp(X.float())

dnn_model_architecture_1 = MLPModel

# we now define everything needed to train the NN (in this case we will only define the optimizer and
# the maximum number of epochs
optimizer_fn = torch.optim.Adam
optimizer_kwargs = {'lr':1e-3}
scheduler_fn = None
scheduler_kwargs = {}
scheduler_config = {}
callback_fns = None
callback_kwargs = None
logger_fn = None
logger_kwargs = {}

# trainer = Trainer(max_epochs=10,
#                  enable_progress_bar=True)

# we can finally define our feature map that encapsulate all those parameters

# depends on dataset
def feature_map(data, context_window_len, max_epochs=10):
    N = data.shape[0]
    trainset_torch = TrajToContextsDataset(data[:int(N*train_val_split)], context_window_len=context_window_len, time_lag=1)
    valset_torch = TrajToContextsDataset(data[int(N*train_val_split):], context_window_len=context_window_len, time_lag=1)

    train_dataloader = DataLoader(trainset_torch)
    val_dataloader = DataLoader(valset_torch)

    dnn_model_kwargs_1 = {
    'input_dim': data.shape[-1],
    'output_dim': 10,
    'hidden_dims': [10, 10],
    'flatten_input': False,
    }
    trainer = Trainer(max_epochs=max_epochs,
                 enable_progress_bar=True)
    feature_map = DPNet(
        encoder=dnn_model_architecture_1,
        optimizer_fn=optimizer_fn,
        trainer=trainer,
        encoder_kwargs=dnn_model_kwargs_1,
        optimizer_kwargs=optimizer_kwargs,
        seed=seed,
        metric_deformation_loss_coefficient=1.,
    )

    feature_map.fit(train_dataloaders=train_dataloader,
                    val_dataloaders= val_dataloader)
    
    return feature_map 