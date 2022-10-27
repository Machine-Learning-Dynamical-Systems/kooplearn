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

def holomorphic_score(feature_map, X, Y, holomorphic_transformation = lambda x: x):
    #To be MAXIMIZED
    T = covariance_matrix(feature_map, X, Y) #Cross covariance
    w = torch.linalg.eigvals(T)
    return holomorphic_transformation(w).sum()

def simplified_VAMP(feature_map, X, Y, p=2):
    #To be MAXIMIZED
    T = covariance_matrix(feature_map, X, Y) #Cross covariance
    w = torch.linalg.svdvals(T)
    return w.pow(p).sum().pow(p**-1)