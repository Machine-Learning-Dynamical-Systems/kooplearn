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
num_features = 10  # number of time series
num_samples_train = 100  # number of time steps
num_samples_val = 100 #
num_samples_test = 100
buffer = 100 # number of time steps to ensure we reach invariant distribution
num_samples = buffer+num_samples_train+num_samples_val+num_samples_test

H = ortho_group.rvs(num_features)
eigs = np.exp(-np.arange(num_features))
A = H @ (eigs * np.eye(num_features)) @ H.T

series = LinearModel(A = A, noise=1.)
data = series.sample(X0 = np.zeros(A.shape[0]), T=num_samples)[buffer:]

# Setting Datasets for learning
trainset = traj_to_contexts(data[:num_samples_train], context_window_len=2, time_lag=1)
valset = traj_to_contexts(data[num_samples_train:-num_samples_test], context_window_len=2, time_lag=1)
testset = traj_to_contexts(data[-num_samples_test:], context_window_len=2, time_lag=1)

trainset_torch = TrajToContextsDataset(data[:num_samples_train], context_window_len=2, time_lag=1)
valset_torch = TrajToContextsDataset(data[num_samples_train:-num_samples_test], context_window_len=2, time_lag=1)

train_dataloader = DataLoader(trainset_torch)
val_dataloader = DataLoader(valset_torch)

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
dnn_model_kwargs_1 = {
    'input_dim': num_features,
    'output_dim': 10,
    'hidden_dims': [10, 10],
    'flatten_input': False,
    }

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

def feature_map(max_epochs=10):
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

# we define the model that encapsulates the feature map and the koopman estimator
# model = DeepEDMD(feature_map=feature_map, rank=100, tikhonov_reg=10)

# model.fit(trainset)

# results = model.predict(valset)


# with open('save.npy', 'wb+') as file:
#     np.save(file, results)