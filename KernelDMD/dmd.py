from scipy.sparse.linalg import eigsh, aslinearoperator
from scipy.linalg import eig, eigh

def DMD(trajectory, kernel, num_modes):
    """DMD using truncated SVD decomposition

    Args:
        trajectory (array): [observations, features]
        kernel (kernel object)
        num_modes (int): number of modes to compute
    """
        Sigma_r, V_r = 
    def evd(self,x, k = 10):      
        return eigsh(aslinearoperator(self.matrix(x,x)), k = k)
    
    def dmd(self,X,r):
        Sigma_r, V_r = self.evd(X[:,:-1].T, k = r)
        Vhat_r = V_r @ np.diag(np.sqrt(Sigma_r))
        Ahat = Vhat_r.T @(self.matrix(X[:,:-1].T,X[:,1:].T) @ Vhat_r)

        evals , evecs = eig(Ahat) 
    
        return evals, Vhat_r@evecs

def DMDs(trajectory, kernel, num_modes):
    """Sparse DMD using Lanczos iterations. Useful for large problems

    Args:
        trajectory (array): [observations, features]
        kernel (kernel object)
        num_modes (int): number of modes to compute
    """
    pass