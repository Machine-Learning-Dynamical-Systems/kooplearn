import numpy as np


__all__ = [
    "grad",
    "grad2"
]


def return_grad(kernel_X, X, Y, diffusion, length_scale):
    r"""
    Compute the first derivative kernel block :math:`N` used in the
    Dirichlet-form estimator of the infinitesimal generator.

    This function evaluates

    .. math::

        N_{i, (k-1)M + j}
        = \langle \phi(x_i), \partial_k \phi(y_j) \rangle,

    where

    - :math:`i = 1,\dots,N`
    - :math:`j = 1,\dots,M`
    - :math:`k = 1,\dots,d`
    - :math:`N` is the number of samples in :math:`X`
    - :math:`M` is the number of samples in :math:`Y`
    - :math:`d` is the dimensionality of the state space

    For an RBF kernel with length-scale :math:`\sigma`, the expression becomes
    (see :footcite:`kostic2024learning`):

    .. math::

        \langle \phi(x_i), \partial_k \phi(y_j) \rangle
        = \sqrt{s_k}\,
          \frac{(x_i - y_j)_k}{\sigma^2}
          \, k(x_i, y_j),

    where :math:`s_k` denotes the diffusion coefficient.

    Parameters
    ----------
    kernel_X : ndarray of shape ``(N, M)``
        Kernel matrix :math:`k(x_i, y_j)`.

    X : ndarray of shape ``(N, d)``
        First set of input samples.

    Y : ndarray of shape ``(M, d)``
        Second set of input samples.

    diffusion : ndarray of shape `(M, d, d)``
        Diffusion tensor :math:`D(x) = b(x)b(x)^\top` used in the generator.
    `

    length_scale : float
        Kernel length-scale :math:`\sigma`.

    Returns
    -------
    N : ndarray of shape ``(N, M d)``
        First derivative kernel block. Columns are ordered as:

        .. math::
            (j = 1,\dots,M) \text{ for } k = 1;
            \quad
            (j = 1,\dots,M) \text{ for } k = 2;
            \quad \dots

    Notes
    -----
    This is the matrix denoted :math:`N` in
    :footcite:`kostic2024learning`.  It is used in the
    Dirichlet-regression operator estimator.

    """

    difference = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    n = X.shape[0]
    m = Y.shape[0]

    d = X.shape[1]
    N = np.zeros((n, m * d))
    sigma = length_scale
    for i in range(n):
        for j in range(m):
            for k in range(0, d):
                N[i, k * m + j] = (
                        diffusion[j, k] * difference[i, j, k] * kernel_X[i, j] / sigma**2
                )
    return N

def return_grad2(kernel_X: np.ndarray, X: np.ndarray, Y, diffusion: np.ndarray,length_scale):
    r"""
    Compute the second derivative kernel block :math:`M` used in the
    Dirichlet-form generator estimator.

    This function evaluates

    .. math::

        M_{ (k-1)N + i,\; (l-1)M + j }
        = \langle \partial_k \phi(x_i), \partial_l \phi(y_j) \rangle,

    where

    - :math:`i = 1,\dots,N`
    - :math:`j = 1,\dots,M`
    - :math:`k,l = 1,\dots,d`

    For an RBF kernel with length-scale :math:`\sigma` (see
    :footcite:`kostic2024learning`):

    .. math::

        \langle \partial_k \phi(x_i), \partial_l \phi(y_j) \rangle
        =
        \begin{cases}
        s_k \left(
            \frac{1}{\sigma^2}
            - \frac{(x_i - y_j)_k^2}{\sigma^4}
        \right)
        k(x_i, y_j),
        & k = l, \\[1.0em]
        - \sqrt{s_k s_l}\,
        \frac{(x_i - y_j)_k (x_i - y_j)_l}{\sigma^4}
        \, k(x_i, y_j),
        & k \neq l.
        \end{cases}

    Parameters
    ----------
    kernel_X : ndarray of shape ``(N, M)``
        Kernel matrix :math:`k(x_i, y_j)`.

    X : ndarray of shape ``(N, d)``
        First set of samples.

    Y : ndarray of shape ``(M, d)``
        Second set of samples.

    diffusion : ndarray of shape `(M, d, d)``
        Diffusion tensor :math:`D(x) = b(x)b(x)^\top` used in the generator.

    length_scale : float
        Kernel length-scale :math:`\sigma`.

    Returns
    -------
    M : ndarray of shape ``(N d, M d)``
        Second derivative block of the kernel. Rows are ordered by
        derivative index :math:`k`, then sample index :math:`i`, i.e.

        .. math::
            ((k-1)N + i) \text{ for } k = 1,\dots,d.

        Columns follow the same ordering for :math:`l,j`.

    Notes
    -----
    This is the matrix called :math:`M` in :footcite:`kostic2024learning`.
    It forms the second part of the Dirichlet-form regression operator.

    """
   
    difference = X[:, np.newaxis, :] - Y[np.newaxis, :, :]

    d = difference.shape[2]
    n = X.shape[0]
    m = Y.shape[0]

    M = np.zeros((n * d, m * d))
    sigma = length_scale
    for i in range(n):
        for j in range(m):
            for k in range(0, d):
                for l in range(0, d):
                    if l == k:
                        M[(k) * n + i, (l) * m + j] = (
                            diffusion[i,k] * diffusion[j,k]
                            * (1 / sigma**2 - difference[i, j, k] ** 2 / sigma**4)
                            * kernel_X[i, j]
                        )
                    else:
                        M[(k) * n + i, (l) * m + j] = (
                                diffusion[i,l] * diffusion[j,l]
                                * (-difference[i, j, k] * difference[i, j, l] / sigma**4)
                                * kernel_X[i, j]
                        )

    return M
