import torch
def GradientDescent(inputs, evolved_inputs, kernel, regularization, learning_rate):
    g = learning_rate
    l = regularization
    K = kernel(inputs)
    evolved_K = kernel(evolved_inputs)
    _c = -2*g*(evolved_K@K)/K.shape[0]
    W = torch.zeros_like(K)
    res = torch.clone(_c)

