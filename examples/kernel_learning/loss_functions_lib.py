import torch

def covariance_matrix(feature_map, X, Y=None):
    #Expecting inputs of shape [num_samples, num_features]
    phi_X = feature_map(X)
    dtype = phi_X.dtype
    dim_inv = torch.as_tensor(X.shape[0]**-1, dtype = dtype)
    if Y is None:
        return dim_inv*torch.matmul(phi_X.transpose(0,1), phi_X)
    else:
        phi_Y = feature_map(Y)
        return dim_inv*torch.matmul(phi_X.transpose(0,1), phi_Y)

def gram_matrix(feature_map, X, Y = None):
    #Expecting inputs of shape [num_samples, num_features]
    if Y is None:
        return covariance_matrix(feature_map, X.transpose(0,1))
    else:
        return covariance_matrix(feature_map, X.transpose(0,1), Y.transpose(0,1))

def kernel_target_alignment(feature_map, X, Y, **kwargs):
    #we expect feature_map to be batch_normalized
    T = covariance_matrix(feature_map, X,Y) #Cross covariance
    C_X = covariance_matrix(feature_map, X) #Input covariance
    C_X_norm = torch.trace(torch.matmul(C_X, C_X))
    G = C_X_norm**-1*torch.matmul(C_X,T)
    phi_X = feature_map(X)
    phi_Y = feature_map(Y)
    #Normalization
    dtype = phi_X.dtype
    dim_inv = torch.as_tensor(X.shape[0]**-1, dtype = dtype)
    return dim_inv*(torch.linalg.matrix_norm(phi_Y - torch.matmul(phi_X, G))**2)

def KRR_loss(feature_map, X, Y, **kwargs):
    T = covariance_matrix(feature_map, X,Y) #Cross covariance
    C_X = covariance_matrix(feature_map, X) #Input covariance
    if 'reg' in kwargs.keys():
        gamma = kwargs['reg'].abs()    
        dtype = gamma.dtype
        device = gamma.device
        dim = C_X.shape[0]
        u = torch.linalg.cholesky(C_X + gamma*torch.eye(dim, dtype=dtype, device=device))
        G = torch.cholesky_solve(T, u)
    else:
        G = torch.matmul(C_X.pinverse(),T)
    phi_X = feature_map(X)
    phi_Y = feature_map(Y)
    #Normalization
    dtype = phi_X.dtype
    dim_inv = torch.as_tensor(X.shape[0]**-1, dtype = dtype)
    return dim_inv*(torch.linalg.matrix_norm(phi_Y - torch.matmul(phi_X, G))**2)

def VAMP(feature_map, X, Y, **kwargs):
    if 'p' in kwargs.keys():
        p = kwargs['p']
    else:
        p=2
    #To be MAXIMIZED
    C_X = covariance_matrix(feature_map, X) #Input covariance
    T = covariance_matrix(feature_map, X, Y) #Cross covariance
    if 'reg' in kwargs.keys():
        gamma = kwargs['reg'].abs()    
        dtype = gamma.dtype
        device = gamma.device
        dim = C_X.shape[0]
        u = torch.linalg.cholesky(C_X + gamma*torch.eye(dim, dtype=dtype, device=device))
        G = torch.cholesky_solve(T, u)
    else:
        G = C_X.pinverse()@T
    return -torch.linalg.svdvals(G).pow(p).sum().pow(p**-1)