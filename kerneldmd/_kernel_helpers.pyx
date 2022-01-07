
# cython: infer_types=True
from libc cimport math
cimport cython
from numpy.math cimport INFINITY
from numpy.math cimport PI
import numpy as np
from cython.parallel cimport prange

ctypedef fused real:
    float
    double
    long double

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pdist2(real[:,:] X1, real[:,:] X2, dtype):
    #Assuming X = [observations, features]
    cdef int n1 = X1.shape[0]
    cdef int n2 = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]
    cdef int d = X1.shape[1]

    cdef Py_ssize_t i, j, k
    cdef real[:,:] D = np.zeros((n1,n2), dtype)

    for i in prange(n1, nogil=True):
        for j in range(n2):
            for k in range(d):
                D[i,j] += math.pow((X1[i,k] - X2[j,k]),2)
    return np.asarray(D)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pdist_sym2(real[:,:] X,  dtype):
    #Assuming X = [observations, features]
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]

    cdef Py_ssize_t i, j, k
    cdef real[:,:] D = np.zeros((n,n), dtype)

    for i in prange(n, nogil=True):
        for j in range(i):
            for k in range(d):
                D[i,j] += math.pow((X[i,k] - X[j,k]),2)
            D[j,i] += D[i,j]
    return np.asarray(D)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pscal(real[:,:] X1, real[:,:] X2, dtype):
    #Assuming X = [observations, features]
    cdef int n1 = X1.shape[0]
    cdef int n2 = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]
    cdef int d = X1.shape[1]

    cdef Py_ssize_t i, j, k
    cdef real[:,:] D = np.zeros((n1,n2), dtype)

    for i in prange(n1, nogil=True):
        for j in range(n2):
            for k in range(d):
                D[i,j] += X1[i,k]*X2[j,k]
    return np.asarray(D)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef pscal_sym(real[:,:] X,  dtype):
    #Assuming X = [observations, features]
    cdef int n = X.shape[0]
    cdef int d = X.shape[1]

    cdef Py_ssize_t i, j, k
    cdef real[:,:] D = np.zeros((n,n), dtype)

    for i in prange(n, nogil=True):
        for j in range(i):
            for k in range(d):
                D[i,j] += X[i,k]*X[j,k]
            D[j,i] += D[i,j]
            
    for i in prange(n, nogil=True):
        for k in range(d):
            D[i,i] += X[i,k]*X[i,k]
    return np.asarray(D)