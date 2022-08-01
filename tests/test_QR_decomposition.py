from sys import path
path.append('../')
from kooplearn.utils import modified_QR
import numpy as np

def test_qr_numpy_euclidean_nopivoting():
    A = np.random.rand(10, 10)
    B = A@A.T + np.eye(10)
    Q, _ = modified_QR(B)
    assert np.allclose(Q.T@Q, np.eye(10))

def test_qr_numpy_euclidean_pivoting():
    A = np.random.rand(10, 5)
    B = A@A.T
    Q, _, _ = modified_QR(B, column_pivoting=True)
    assert Q.shape[1] <= 5

def test_qr_numpy_weighted_nopivoting():
    A = np.random.rand(10, 10)
    M = np.random.rand(10,10)
    M = M@M.T
    B = A@A.T
    Q, _ = modified_QR(B, M=M)
    assert np.allclose(Q.T@M@Q, np.eye(10))