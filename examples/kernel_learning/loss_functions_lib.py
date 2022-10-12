import torch

def covariance_matrix(feature_map, X, Y=None):
    #Expecting inputs of shape [num_samples, num_features]
    phi_X = feature_map(X)
    dtype = phi_X.type()
    dim_inv = torch.as_tensor(X.shape[0]**-1, dtype = dtype)
    if Y is None:
        return dim_inv*torch.matmul(phi_X.transpose(), phi_X)
    else:
        phi_Y = feature_map(Y)
        return dim_inv*torch.matmul(phi_X.transpose(), phi_Y)

def gram_matrix(feature_map, X, Y = None):
    #Expecting inputs of shape [num_samples, num_features]
    if Y is None:
        return covariance_matrix(feature_map, X.transpose())
    else:
        return covariance_matrix(feature_map, X.transpose(), Y.transpose())


def VAMP_score():
    pass


L = torch.linalg.cholesky(C_reg) #C_reg = L@L.T [d_f, d_f]
C = torch.cholesky_solve(phi_X.T, L) # [d_f, n]