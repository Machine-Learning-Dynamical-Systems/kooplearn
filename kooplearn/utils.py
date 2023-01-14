import numpy as np

from scipy.sparse.linalg import aslinearoperator, LinearOperator

from sklearn.utils import check_array, check_random_state

from warnings import warn

def sort_and_crop(vec, num_components = None):
    """Return the i

    Args:
        vec (ndarray): 1D array of floats
        num_component (int, optional): Number of indices to retain. Defaults to None corresponding to every indices.

    Returns:
        ndarray: array of integers corresponding to the indices of the largest num_components elements in vec.
    """
    assert np.ndim(vec) == 1, "'vec' must be a 1D array"
    sort_perm = np.argsort(vec)[::-1] # descending order
    if num_components is None:
        return sort_perm
    else:
        return sort_perm[:num_components]
    
def weighted_norm(A, M = None):
    """Weighted norm of the columns of A.

    Args:
        A (ndarray): 1D or 2D array. If 2D, the columns are treated as vectors.
        M (ndarray or LinearOperator, optional): Weigthing matrix. the norm of the vector a is given by a.T@M@a. Defaults to None, corresponding to the Identity matrix. Warning: no checks are performed on M being a PSD operator. 

    Returns:
        (ndarray or float): if A.ndim == 2 returns 1D array of floats corresponding to the norms of the columns of A. Else return a float.
    """    
    assert A.ndim <= 2, "'A' must be a vector or a 2D array"
    _1D = (A.ndim == 1)
    if M is None:
        norm = np.linalg.norm(A, axis=0)
    else:
        if _1D:
            _A = aslinearoperator(M).matvec(np.ascontiguousarray(A))
        else:
            _A = aslinearoperator(M).matmat(np.asfortranarray(A))  
        norm = np.sqrt(
                np.real(
                    np.sum(
                        np.conj(A)*_A, 
                        axis=0)
                    )
                )
    return norm

def weighted_dot_product(A, B, M=None):
    """Weighted dot product between the columns of A and B. The output will be equivalent to np.conj(A).T@M@B

    Args:
        A, B (ndarray): 1D or 2D arrays.
        M (ndarray or LinearOperator, optional): Weigthing matrix. Defaults to None, corresponding to the Identity matrix. Warning: no checks are performed on M being a PSD operator. 

    Returns:
        (ndarray or float): the result of np.conj(A).T@M@B.
    """    
    assert A.ndim <= 2, "'A' must be a vector or a 2D array"
    assert B.ndim <= 2, "'B' must be a vector or a 2D array"
    A_adj = np.conj(A.T)
    _B_1D = (B.ndim == 1)
    if M is None:
        return np.dot(A_adj, B)
    else:
        if _B_1D:
            _B = aslinearoperator(M).matvec(np.ascontiguousarray(B))
        else:
            _B = aslinearoperator(M).matmat(np.asfortranarray(B))
        return np.dot(A_adj, _B)

def _column_pivot(Q, R, k, squared_norms, columns_permutation):
    """
        Helper function to perform column pivoting on the QR decomposition at the k iteration. No checks are performed. For internal use only.
    """
    _arg_max = np.argmax(squared_norms[k:])
    j = k + _arg_max
    _in = [k, j]
    _swap = [j,k]
    #Column pivoting
    columns_permutation[_in] = columns_permutation[_swap]
    Q[:, _in] = Q[:,_swap]
    R[:k, _in] = R[:k,_swap]
    squared_norms[_in] = squared_norms[_swap]
    return Q, R, squared_norms, columns_permutation

def modified_QR(A, M = None, column_pivoting = False, rtol = 2.2e-16, verbose = False):
    """Modified QR algorithm with column pivoting. Implementation follows the algorithm described in [1].

    Args:
        A (ndarray): 2D array whose columns are vectors to be orthogonalized.
        M (ndarray or LinearOperator, optional): PSD linear operator. If not None, the vectors are orthonormalized with respect to the scalar product induced by M. Defaults to None corresponding to Identity matrix.
        column_pivoting (bool, optional): Whether column pivoting is performed. Defaults to False.
        rtol (float, optional): relative tolerance in determining the numerical rank of A. Defaults to 2.2e-16. This parameter is used only when column_pivoting == True.
        verbose (bool, optional): Whether to print informations and warnings about the progress of the algorithm. Defaults to False.

    Returns:
        Q, R: the matrices Q and R satisfying A = QR. If column_pivoting is True, the permutation of the columns of A is returned as well.
    
    [1] A. Dax: 'A modified Gram–Schmidt algorithm with iterative orthogonalization and column pivoting', https://doi.org/10.1016/S0024-3795(00)00022-7. 
    """    
    A = check_array(A) #Ensure A is non-empty 2D array containing only finite values.
    num_vecs = A.shape[1]
    effective_rank = num_vecs
    dtype = A.dtype
    Q = np.copy(A)
    R = np.zeros((num_vecs,num_vecs), dtype=dtype)

    _roundoff = 1e-8 #From reference paper
    _tau = 1e-2 #From reference paper

    if column_pivoting: #Initialize variables for fast pivoting, without re-evaluation of the norm at each step.
        squared_norms = weighted_norm(Q, M = M)**2
        max_norm = np.sqrt(np.max(squared_norms))
        columns_permutation = np.arange(num_vecs)

    for k in range(num_vecs):
        if column_pivoting:
            Q, R, squared_norms, columns_permutation = _column_pivot(Q, R, k, squared_norms, columns_permutation)
            norms_error_estimate = squared_norms * _roundoff
        if k != 0: #Reorthogonalization of the column k+1 of A with respect to the previous orthonormal k vectors.     
            alpha = weighted_dot_product(Q[:,:k], Q[:,k], M=M) #alpha = Q[:,:k].T@M@Q[:,k]
            R[:k,k] += alpha
            Q[:,k] -= np.dot(Q[:,:k],alpha)

        #Numerical rank detection, performed only when column_pivoting == True
        norm_at_iter_k = weighted_norm(Q[:,k], M=M)
        if column_pivoting:
            if norm_at_iter_k < rtol*max_norm:
                effective_rank = k
                if verbose:
                    warn("Numerical rank of A has been reached with a relative tolerance rtol = {:.2e}. Effective rank = {}. Stopping Orthogonalization procedure.".format(rtol, effective_rank))
                break    
        # Normalization of the column k + 1 
        R[k, k] = norm_at_iter_k
        Q[:, k] = Q[:, k] / R[k, k]
        # Orthogonalization of the remaining columns with respect to Q[:,k], i.e. the k+1 column of Q.
        if k < num_vecs - 1:
            R[k,k+1:] = weighted_dot_product(Q[:, k+1:], Q[:, k], M=M)       
            Q[:,k+1:] -= np.outer(Q[:, k], R[k,k+1:])
            if column_pivoting: #Try fast update of the squared norms, recompute if numerical criteria are not attained.
                squared_norms[k+1:] -= R[k,k+1:]**2 #Update norms using Phythagorean Theorem
                update_error_mask = _tau*squared_norms[k+1:] < norms_error_estimate[k+1:] #Check if the error estimate is too large
                if any(update_error_mask):
                    squared_norms[k+1:][update_error_mask] = weighted_norm(Q[:,k+1:][:,update_error_mask], M=M) #Recompute the norms if necessary.
    if column_pivoting:
        return Q[:,:effective_rank], R[:effective_rank], columns_permutation[:effective_rank]
    else:
        return Q[:,:effective_rank], R[:effective_rank]

class SquaredKernel(LinearOperator):
    """KernelSquared:helper class to repeatedly apply alpha*K@K+beta*K.
    """
    def __init__(self, K, alpha, beta):
        self.K = K
        self.is_linop = False
        if isinstance(K, LinearOperator):
            self.is_linop = True
        else:
            self.M = (alpha*K + beta)@K
        self.dtype = K.dtype #Needed by LinearOperator superclass
        self.shape = K.shape #Needed by LinearOperator superclass
        self.alpha = alpha
        self.beta = beta

    def _matvec(self, x):
        if self.is_linop:
            v = np.ascontiguousarray(self.K @ x)
            return self.alpha * self.K @ v + self.beta * v
        else:
            return self.M@x

def randomized_range_finder(A_A_adj, size, n_iter, power_iteration_normalizer="none", M=None, random_state=None):
    """Randomized range finder. Adapted from sklearn.utils.extmath.randomized_range_finder

    Args:
        A_A_adj (ndarray, LinearOperator): Object representing A@A_adj, where A is the matrix whose range is to be found.
        size (int): Number of vectors to be generated and used to find the range of A.
        n_iter (int): Number of power iterations to be used to estimate the matrix range.
        power_iteration_normalizer (str, optional): Scheme to normalize the power iterations at each step. Defaults to "none". Available options are: "QR" and "column_pivoted_QR".
        M (ndarray or LinearOperator, optional): PSD linear operator. If not None, the vectors are orthonormalized with respect to the scalar product induced by M. Defaults to None corresponding to Identity matrix, i.e. standard Euclidean norm.
        random_state (int, optional):  int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data, i.e. getting the random vectors to initialize the algorithm.
        Pass an int for reproducible results across multiple function calls.

    Returns:
        ndarray: An (A_A_adj.shape[0], size) matrix the range of which approximates the range of A.
    """    
    #Adapted from sklearn.utils.extmath.randomized_range_finder
    A = aslinearoperator(A_A_adj)
    random_state = check_random_state(random_state)
    # Generating normal random vectors with shape: (A_A_adj.shape[0], size)
    assert size > 0, "Size must be greater than zero"
    Q = random_state.normal(size=(A_A_adj.shape[0], size))
    if hasattr(A_A_adj, "dtype") and A_A_adj.dtype.kind == "f":
        # Ensure f32 is preserved as f32
        Q = Q.astype(A_A_adj.dtype, copy=False)
    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for _ in range(n_iter):
        if power_iteration_normalizer == "none":
            Q = A.matmat(Q)
        elif power_iteration_normalizer == "QR":
            Q, _ = modified_QR(A.matmat(Q), M=M)
        elif power_iteration_normalizer == "column_pivoted_QR":
            Q, _, _ = modified_QR(A.matmat(Q), M=M, column_pivoting=True)
    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _, _ = modified_QR(A.matmat(Q), M=M, column_pivoting=True)
    return Q