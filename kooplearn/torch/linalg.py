import torch
from kooplearn.torch.typing import RealLinalgDecomposition

def generalized_eigh(A: torch.Tensor, B: torch.Tensor) -> tuple:
     #A workaround to solve a real symmetric GEP Av = \lambda Bv problem in JAX. (!! Not numerically efficient)
     Lambda, Q = torch.linalg.eigh(B)
     rsqrt_Lambda = torch.diag(Lambda.rsqrt())
     sqrt_B = Q@rsqrt_Lambda
     _A = 0.5*(sqrt_B.T@(A@sqrt_B) + sqrt_B.T@((A.T)@sqrt_B)) #Force Symmetrization
     values, _tmp_vecs = torch.linalg.eigh(_A) 
     vectors = Q@(rsqrt_Lambda@_tmp_vecs)
     return RealLinalgDecomposition(values, vectors)

def spd_norm(vecs: torch.Tensor, spd_matrix: torch.Tensor) -> torch.Tensor:
     _v = torch.mm(spd_matrix, vecs)
     _v_T = torch.mm(spd_matrix.T, vecs)
     return torch.sqrt(0.5*torch.linalg.vecdot(vecs, _v + _v_T, dim = 0).real)