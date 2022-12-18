#IMPORTS
#STDLIB

#ESSENTIALS
import numpy as np

#MISC
from kooplearn.kernels import Kernel
def error_decay_variable(n: int, delta: float, kernel_sup_norm: float = 1.0) -> float:
    return (n**-1)*kernel_sup_norm*np.log(np.power(n, 2)*8*(delta**-1))

def hat_eps_n(X: np.ndarray, kernel: Kernel, delta: float, kernel_sup_norm: float = 1.0) -> float:
    n = X.shape[0]
    J = error_decay_variable(n, delta, kernel_sup_norm=kernel_sup_norm)
    cov_norm = np.linalg.norm((n**-1)*kernel(X), ord=2) #Operator norm of the Gram Matrix
    return 91.0*J + 6.0*np.sqrt(cov_norm*J)

def eps_n_1(X: np.ndarray, delta:float, tikhonov_reg:float, kernel_sup_norm: float = 1.0) -> float:
    n = X.shape[0]
    J = error_decay_variable(n, delta, kernel_sup_norm=kernel_sup_norm)
    return 12.0*(tikhonov_reg**-1)*J + 6.0*np.sqrt((tikhonov_reg**-1)*J)

def hat_eps_n_2(X: np.ndarray, kernel: Kernel, delta:float, tikhonov_reg:float, kernel_sup_norm: float = 1.0) -> float:
    n = X.shape[0]
    J = error_decay_variable(n, delta, kernel_sup_norm=kernel_sup_norm)
    cov_norm = np.linalg.norm((n**-1)*kernel(X), ord=2) #Operator norm of the Gram Matrix
    return 12.0*(tikhonov_reg**-0.5)*J + 17.0*(tikhonov_reg**-0.25)*J*(kernel_sup_norm**0.5) + 6.0*np.sqrt((tikhonov_reg**-0.25)*J*(cov_norm**0.5))

def eigenvalue_errors(
    eta: np.ndarray,
    sval_B_rp1: float, 
    X: np.ndarray, 
    kernel: Kernel, 
    delta:float, 
    tikhonov_reg:float, 
    kernel_sup_norm: float = 1.0
    ) -> np.ndarray:
    _d = 0.25*delta
    E_n_1 = 1.0 - eps_n_1(X, _d, tikhonov_reg, kernel_sup_norm=kernel_sup_norm)
    E_n_2 = 2.0*hat_eps_n_2(X, kernel, _d, tikhonov_reg,kernel_sup_norm=kernel_sup_norm)

    if E_n_1 < 0:
        _err = np.empty_like(eta)
        _err[:] = np.nan
        return _err
    else:
        numerator = np.sqrt(tikhonov_reg)*E_n_1 + E_n_2 + sval_B_rp1*np.sqrt(E_n_1)
        denumerator = E_n_1*np.sqrt(eta - tikhonov_reg - hat_eps_n(X, kernel, _d, kernel_sup_norm=kernel_sup_norm))

        return numerator*(denumerator**-1)

