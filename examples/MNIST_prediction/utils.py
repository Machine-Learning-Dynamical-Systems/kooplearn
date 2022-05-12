#Misc
import signal
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
from tqdm import tqdm
from scipy.special import softmax
#DynamicalSystems
from DynamicalSystems.kernels import Kernel

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

class GracefulExiter():
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("Exit flag set to True (repeat to exit now)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state
class DeepKernel(Kernel):
    def __init__(self, net, softmax=False):
        self.softmax = softmax
        cpu = torch.device('cpu')
        state_dict = net.state_dict()
        self._feature_map = net.__class__().to(cpu)
        self._feature_map.load_state_dict(state_dict)
    def __call__(self, X, Y=None, backend='auto'):
        _d = int(np.sqrt(X.shape[1]))
        if backend == 'keops':
            raise NotImplementedError("KeOps backend is not implemented for DeepKernel.")
        else:
            with torch.no_grad():
                _X = torch.from_numpy(X.reshape(-1, _d, _d))[:,None,:,:]
                Phi_X = self._feature_map(_X).numpy()
                if Y is None:
                    Phi_Y = Phi_X.copy()
                else:
                    _Y = torch.from_numpy(Y.reshape(-1, _d, _d))[:,None,:,:]
                    Phi_Y = self._feature_map(_Y).numpy()
                if self.softmax:
                    Phi_X = softmax(Phi_X, axis=1)
                    Phi_Y = softmax(Phi_Y, axis=1)
                return Phi_X@(Phi_Y.T) 

def risk(X, Y, reg):
    n, d = X.shape 
    C_X = torch.mm(X.T, X)*(n**-1)
    C_XY = torch.mm(X.T,Y)*(n**-1)
    evals_X = torch.linalg.eigvalsh(C_X)
    if reg is not None:
        tikhonov = torch.eye(d, dtype=X.dtype, device=X.device)*(reg)
        L = torch.linalg.cholesky(C_X + tikhonov) #K_reg = L@L.T
        C = torch.cholesky_solve(C_XY,L)
    else:
        p = torch.pinverse(C_X)
        C = p@C_XY
    V = Y.T - C.T@X.T
    norm_XY = torch.linalg.matrix_norm(C_XY, ord=2)
    norm_X = torch.linalg.matrix_norm(C_X, ord=2)
    
    return (norm_XY, norm_X, evals_X), (n**-1)*torch.linalg.matrix_norm(V, ord = 2)**2 

#Define training loop
def train(data, net, optimizer, params, reg, norm_reg = 1, scheduler = None, split_size = 0):
    exiter = GracefulExiter()
    if split_size > 0:
        _X = torch.split(data[:-1], split_size)
        _Y = torch.split(data[1:], split_size)
        inputs = [(_X[i].to(device), _Y[i].to(device)) for i in range(len(_X))]
    else:
        inputs = [(data[:-1].to(device), data[1:].to(device))]
    epochs = params['epochs']
    lss = "N/A"
    progress = tqdm(range(epochs), desc=f"Loss: {lss}", unit='Epoch')
    for idx in progress:
        for X, Y in inputs:
            norm = torch.linalg.norm(nn.utils.parameters_to_vector(net.parameters()))**2
            optimizer.zero_grad()
            out_X = net.forward(X)
            out_Y = net.forward(Y)
            cov_norm, loss = risk(out_X, out_Y, reg)
            loss += norm_reg*norm 
            loss.backward()
            optimizer.step()
            net._loss_history.append(loss.item())
            net._norm_history.append((cov_norm[0].item(), cov_norm[1].item(),norm.item()))
            net._evals_history.append(cov_norm[2])
        if idx%10 == 0:
            if scheduler is not None:
                scheduler.step()
            progress.set_description("Loss: " + "{:.2e}".format(loss.item()) + " | Norm: " + "{:.2e}".format(norm.item()))
        if loss.item() <= 1e-15:
            break
        if exiter.exit():
            break
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