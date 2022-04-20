
from cmath import sqrt
from genericpath import exists
from matplotlib.pyplot import axis
from scipy.sparse.linalg import eigs, eigsh, aslinearoperator, LinearOperator, cg
from scipy.linalg import eig, eigh, solve
from scipy.sparse import diags
from pykeops.numpy import Vi
from DynamicalSystems.utils import modified_QR, parse_backend, _check_real, IterInv, rSVD
import numpy as np
from warnings import warn

class KoopmanRegression:
    def __init__(self, data, evolved_data, kernel, regularizer = None, rank=None, center_kernel=False, backend='auto', rank_reduction='r3', random_state = 0, powers = 2, offset = None, tol = 1e-6, ):
        self.backend = parse_backend(backend, data)
        self.kernel = kernel
        self.X, self.Y = data, evolved_data
        self.rank = rank
        self.reg = regularizer
        self.center_kernel = center_kernel
        self.rank_reduction = rank_reduction
        self.n_samples = self.X.shape[0] 
        self.random_state = random_state
        self.powers = powers 
        self.offset = offset
        self.tol = tol
        self.modes_ = None
        self.evals = None

        if self.backend == 'keops':
            self.K_X = aslinearoperator(self.kernel(self.X, backend=self.backend))
            self.K_Y = aslinearoperator(self.kernel(self.Y, backend=self.backend))
            self.K_YX = aslinearoperator(self.kernel(self.Y, self.X, backend=self.backend))
        else:
            self.K_X = self.kernel(self.X, backend=self.backend)
            self.K_Y = self.kernel(self.Y, backend=self.backend)
            self.K_YX = self.kernel(self.Y, self.X, backend=self.backend)
        
        self.dtype = self.K_X.dtype
        
        if center_kernel:
            dK_Y = _center_kernel(self.kernel, self.Y, self.Y, self.X, averaged_indices=(True, True), backend=self.backend)
            dK_YX = _center_kernel(self.kernel, self.Y, self.X, self.X, averaged_indices=(True, False), backend=self.backend)
            if self.backend == 'keops':
                self.K_Y += dK_Y
                self.K_YX += dK_YX
            else:
                Id = np.eye(self.X.shape[0], dtype = self.X.dtype)
                self.K_Y += dK_Y.matmat(Id)
                self.K_YX += dK_YX.matmat(Id)
        
        if self.rank is not None:
            if rank_reduction == 'r3':
                print("INIT: Computing low-rank-projection via Reduced Rank Regression")
                self.V, self.U = self.r3()
            elif rank_reduction == 'pcr':
                print("INIT: Computing low-rank-projection via Principal Component Regression")
                self.V, self.U = self.pcr()
            elif rank_reduction == 'r4':
                print("INIT: Computing low-rank-projection via Randomized Reduced Rank Regression")
                self.V, self.U = self.r4()
            else:
                raise ValueError(f"Unrecognized backend '{rank_reduction}'. Accepted values are 'auto', 'r3', 'r4' or 'pcr'.")
        else:
            if rank_reduction == 'r4':
                print("INIT: Computing low-rank-projection via Randomized Reduced Rank Regression")
                self.V, self.U = self.r4()
            else:
                pass
        # print("INIT: Computing low-rank-projection")
        # self.V, self.U = self._low_rank_projector()

    def pcr(self):
        if self.reg is not None:
            if self.backend == 'keops':
                tikhonov = aslinearoperator(diags(np.ones(self.n_samples, dtype=self.dtype)*(self.reg*self.n_samples)))
                S, V = eigs(self.K_X + tikhonov, self.rank)
                self.sigma_sq = S**2
                sort_perm = np.argsort(self.sigma_sq)[::-1]
                self.sigma_sq = self.sigma_sq[sort_perm]
                V = V[:,sort_perm]
                S = S[sort_perm]
            else:
                tikhonov = np.eye(self.n_samples, dtype=self.dtype)*(self.reg*self.n_samples)
                S, V = eigh(self.K_X + tikhonov)
                S = S[::-1][:self.rank]
                V = V[:,::-1][:,:self.rank]
        else:
            if self.backend == 'keops':
                S, V = eigs(self.K_X, self.rank)
                self.sigma_sq = S**2
                sort_perm = np.argsort(self.sigma_sq)[::-1]
                self.sigma_sq = self.sigma_sq[sort_perm]
                V = V[:,sort_perm]
                S = S[sort_perm]
            else:
                S, V = eigh(self.K_X)
                S = S[::-1][:self.rank]
                V = V[:,::-1][:,:self.rank]
        
        self.sigma_sq = S**2
        return V * np.sqrt(self.n_samples) , V@np.diag(1./S) * np.sqrt(self.n_samples)
        #return V  , V@np.diag(1./S) 

        #S, V = eigs(self.K_X + np.eye(self.n_samples, dtype=self.K_X.dtype)*(self.reg*self.n_samples), self.rank)
        #self.sigma_sq = S**2
        #return V * np.sqrt(self.n_samples) , V@np.diag(1./S) * np.sqrt(self.n_samples)
        #self.U_dmd = V@np.diag(1./S)
        #self.V_dmd = V


        # C = self.K_YX@self.U_dmd 
        # vals, lv, rv =  scipy.linalg.eig(self.V_dmd.T@C, left=True, right=True)
        # self.evals_dmd = vals 
        # self.levecs_dmd = self.V_dmd@lv
        # self.revecs_dmd = self.U_dmd@rv
        # error_ = np.linalg.norm(self.K_YX @ self.revecs_dmd - self.K_X @ self.revecs_dmd@np.diag(self.evals_dmd),ord = 2,axis=0)
        # print(f"Error on eigenfunctions is {error_}") 
        
    def r3(self):
        #For the moment data_kernel = LinearOperator, evolved_data_kernel = LinearOperator
        dim = self.n_samples
        inverse_dim = self.n_samples**-1
        
        K = inverse_dim*(self.K_Y@self.K_X) 
        K.dtype = self.K_X.dtype
        
        if self.reg is not None:
            if self.backend == 'keops':
                tikhonov = aslinearoperator(diags(np.ones(dim, dtype=K.dtype)*(self.reg*dim)))
                alpha =  self.reg*dim
                Minv = IterInv(self.kernel, self.X, alpha)
                sigma_sq, U = eigs(K, self.rank, self.K_X + tikhonov,  Minv=Minv)
            else:
                tikhonov = np.eye(dim, dtype=K.dtype)*(self.reg*dim)
                sigma_sq, U = eig(K, self.K_X + tikhonov)
        
                assert np.max(np.abs(np.imag(sigma_sq))) < 1e-8
                sigma_sq = np.real(sigma_sq)
                sort_perm = np.argsort(sigma_sq)[::-1]
                #print(f'Error compared to full rank is {sigma_sq[sort_perm][self.rank:].sum()}')
                sigma_sq = sigma_sq[sort_perm][:self.rank]
                #self.sigma_sq = sigma_sq[::-1][:self.rank]
                #print(f'Sigma^2 are {sigma_sq}')
                U = U[:,sort_perm][:,:self.rank]
                #U = U[:,::-1][:,:self.rank]

            #Check that the eigenvectors are real (or have a global phase at most)
            if not _check_real(U):
                U_global_phase_norm = np.angle(U).std(axis=0)
                if U_global_phase_norm.mean()  > 1e-8:
                    raise ValueError("Computed projector is not real. The kernel function is either severely ill conditioned or non-symmetric")
                else:
                    #It has a global complex phase, take absolute.
                    U = np.real(U@np.diag(U_global_phase_norm)) ## WRONG
            else:
                U = np.real(U)
            
            U = modified_QR(U, self.backend, inverse_dim*(self.K_X@(self.K_X + tikhonov)))
            V = (self.K_X@np.asfortranarray(U))            
        else:
            if self.backend == 'keops':
                raise ValueError(f"Unsupported backend '{self.backend}' without the regularization.")
            else:
                sigma_sq, V = eigs(self.K_Y, self.rank)

                if not _check_real(V):
                    V_global_phase_norm = np.angle(V).std(axis=0).mean()
                    if V_global_phase_norm  > 1e-8:
                        raise ValueError("Computed projector is not real. The kernel function is either severely ill conditioned or non-symmetric")
                    else:
                        #It has a global complex phase, take absolute.
                        V = np.real(V-V_global_phase_norm)
                else:
                    V = np.real(V)

                V = V@np.diag(np.sqrt(dim)/(np.linalg.norm(V,ord=2,axis=0)))
                U = solve(self.K_X, V, assume_a='sym')
                # U = np.empty_like(V)
                # for k in range(self.rank):
                #     U[:,k], _ = cg(self.K_X, V[:,k])

        self.sigma_sq = sigma_sq
        return V, U


    def r4(self):
        if self.backend == 'keops':
            raise ValueError(f"Unsupported backend '{self.backend}' for Randomized Reduced Rank Regression.")

        if self.reg is None:
            raise ValueError(f"Unsupported unregularized Randomized Reduced Rank Regression.")

        if self.rank is None:
            self.rank = int(np.trace(self.K_Y)/np.linalg.norm(self.K_Y,ord =2))+1
            print(f'Numerical rank of the output kernel is approximatly {int(self.rank)} which is used.')

        if self.offset is None:
            self.offset = 2*self.rank

        l = self.rank + self.offset
        Omega = np.random.randn(self.n_samples,l)
        Omega = Omega @ np.diag(1/np.linalg.norm(Omega,axis=0))
        for j in range(self.powers):
            KyO = self.K_Y@Omega
            Omega = KyO - self.n_samples*self.reg*solve(self.K_X+self.n_samples*self.reg*np.eye(self.n_samples),KyO,assume_a='pos')
        KyO = self.K_Y@Omega
        Omega = solve(self.K_X+self.n_samples*self.reg*np.eye(self.n_samples), KyO, assume_a='pos')
        Q = modified_QR(Omega, backend = self.backend, M = self.K_X@self.K_X/self.n_samples+self.K_X*self.reg)
        if Q.shape[1]<self.rank:
            print(f"Actual rank is smaller! Detected rank is {Q.shape[1]} and is used.")   
        C = self.K_X@Q
        sigma_sq, evecs = eigh((C.T @ self.K_Y) @ C)
        sigma_sq = sigma_sq[::-1][:self.rank]/(self.n_samples**2)
        evecs = evecs[:,::-1][:,:self.rank]
        
        U = Q @ evecs
        V = self.K_X @ U
        error_ = np.linalg.norm(self.K_X@V/self.n_samples - (V+self.n_samples*self.reg*U)@np.diag(sigma_sq),ord=1)
        if  error_> 1e-6:
            print(f"Attention! l1 Error in GEP is {error_}")
        self.sigma_sq = sigma_sq
        self.rank = sigma_sq.shape[0]

        return V, U
    
    def eig(self, X=None):
        dim = self.K_X.shape[0]
        dim_inv = dim**(-1)
        if self.reg is None and self.rank is None:
            raise ValueError(f"Unsupported eigenvalue computation in full rank and without regularization.")

        if self.rank is not None:
            if self.backend == 'keops':
                C = dim_inv* self.K_YX.matmat(np.asfortranarray(self.U)) 
            else:
                C = dim_inv* self.K_YX@self.U 
            vals, lv, rv =  eig(self.V.T@C, left=True, right=True)

            if X is not None:
                Kr = self.kernel(X, self.X, backend=self.backend)
                Kl = self.kernel(X, self.X, backend=self.backend)
                if self.center_kernel:
                    warn("The left eigenfunctions are evaluated with the standard kernel, i.e. without centering.")
                if self.backend == 'keops':
                    self.evals = vals
                    self.levecs = aslinearoperator(Kl).matmat(lv)
                    self.revecs = aslinearoperator(Kr).matmat(rv)
                else:
                    self.evals = vals 
                    self.levecs = Kl@lv
                    self.revecs =  Kr@rv
            else:
                self.evals = vals 
                self.levecs = self.V@lv
                self.revecs = self.U@rv
        else:
            tikhonov = np.eye(dim, dtype=self.K_X.dtype)*(self.reg*dim)
            C = np.linalg.solve(self.K_X+tikhonov, self.K_YX) 

            vals, lv, rv =  eig(C, left=True, right=True)
            self.evals = vals 
            self.revecs = rv
            self.levecs = lv
        #normalize_ = self.evals.shape[0]
        #self.revecs = self.revecs @ np.diag(1/np.sqrt(normalize_*np.diag(self.revecs.T@(self.K_X@self.revecs))))
        #self.levecs = self.levecs @ np.diag(1/np.sqrt(normalize_*np.diag(self.levecs.T@(self.K_Y@self.levecs))))

        return self.evals, self.levecs, self.revecs
    
    def spectral_error(self, data = None):
        if data is None:
            KX = self.K_X
            KYX = self.K_YX
            normalize_ = self.n_samples
        else:
            normalize_ = np.sqrt(self.n_samples*data[0].shape[0]) 
            KX = self.kernel(data[0], self.X, backend=self.backend)
            KYX = self.kernel(data[1], self.X, backend=self.backend)

        return np.linalg.norm(KYX @ self.revecs - KX @ self.revecs@np.diag(self.evals), ord = 2, axis=0) /normalize_

    def risk(self, data = None):
        if data is None:      
            # risk = np.trace(self.K_Y)
            # if self.rank is None:
            #     risk -= 1
            #     risk = risk /self.n_samples 
            # else:
            #     C = (self.K_Y  @ self.V).T @ (self.K_X @ self.U) 
            #     risk -= np.trace(C) / self.n_samples
            #     if self.reg is not None:
            #         risk += np.trace(C@(self.V.T@self.U)) *self.reg / self.n_samples
            #     risk = risk /self.n_samples
            normalize_ = self.n_samples
            risk = np.trace(self.K_Y)
            if self.rank is None:
                risk -= 1
                risk = risk /self.n_samples 
            else:
                U1 = self.K_X @ self.U
                V1 = self.K_Y  @ self.V
                C = self.V.T @ V1 
                risk -= 2*np.trace(V1.T@U1) / normalize_
                risk += np.trace((U1.T@U1) @ C)  / normalize_**2
                risk = risk / normalize_  
        else:
            normalize_ = np.sqrt(self.n_samples*data[0].shape[0]) 
            risk = np.trace(self.kernel(data[1], data[1], backend=self.backend))
            if self.rank is None:
                risk -= 1
                risk = risk /self.n_samples 
            else:
                U1 = self.kernel(data[0],self.X, backend=self.backend) @ self.U
                V1 = self.kernel(data[1], self.Y, backend=self.backend)  @ self.V
                C = self.V.T @ (self.K_Y @ self.V) 
                risk -= 2*np.trace(V1.T@U1) / normalize_
                risk += np.trace((U1.T@U1) @ C)  / normalize_**2
                risk = risk / normalize_ 

        return risk

    def modes(self, f = None):
        assert self.rank is not None, "Rank must be specified to compute modes"
        if f is not None:
            observable = f(self.X)
        else:
            observable = self.X   
        U_tilde = self.revecs
        if self.backend == 'keops':
            F = self.U.T.matmat(np.asfortranarray(observable))
            D = (self.U.T)@self.K_X.matmat(np.asfortranarray(U_tilde))
        else:
            F = self.U.T@observable
            D = (self.U.T)@self.K_X@U_tilde
        self.modes_ = solve(D, F)
        return self.modes_


    def __init_forecast(self):
        if self.rank == None:
            dim = self.K_X.shape[0]
            dt = self.K_X.dtype
            if self.backend == 'keops':     
                alpha = self.reg*dim
                M = self.kernel(self.X, backend=self.backend)
                _X = Vi(self.Y)
                self._forecast_Z =M.solve(_X, alpha = alpha, eps = 1e-6)
            else:
                tikhonov = np.eye(dim, dtype=dt)*(self.reg*dim)
                K_reg = self.K_X + tikhonov
                self._forecast_Z = solve(K_reg,self.Y, assume_a='pos') 
        else:
                self._forecast_Z = (self.V.T)@self.Y

    def forecast(self, initial_point):
        try:
            if self.rank == None:
                _forecast_S = self.kernel(initial_point[np.newaxis, :], self.X, backend = 'cpu')
            else:
                _init_k = self.kernel(initial_point[np.newaxis, :], self.X, backend = 'cpu')
                if self.backend == 'keops':
                    _forecast_S = aslinearoperator(_init_k).matmat(self.U)
                else:
                    _forecast_S = _init_k@self.U
            return np.squeeze(_forecast_S@self._forecast_Z)
        except AttributeError:
            self.__init_forecast()
            return self.forecast(initial_point)

    def modal_decompostion(self, f = None, time_step = 1, which = None):
        
        dim = self.X.shape[1]
        n = self.X.shape[0]

        if self.modes_ is None:
            self.modes(f)

        if which is not None:
            evals = self.evals[which]
            revecs = self.revecs[:,which]
            if len(self.modes_.shape) == 1:
                modes_ = self.modes_[which]
            else:
                modes_ = self.modes_[which,:]
        else:
            evals = self.evals
            revecs = self.revecs
            modes_ = self.modes_
        omegas = np.log(evals)/time_step
        evolve = lambda x0, t: np.real((np.exp(t[np.newaxis].T @ omegas[np.newaxis]) @ np.diag((self.kernel(x0[np.newaxis],self.X,backend=self.backend) @ revecs)[0]) ) @ modes_)
        return evolve


def _center_kernel(kernel, X, Y, D, averaged_indices, backend):
    K_Y = kernel(Y, D, backend=backend).sum(1).squeeze() #Vector
    K_X = kernel(X, D, backend=backend).sum(1).squeeze() #Vector
    K_D = kernel(D, D, backend=backend).sum(0).squeeze().sum(0) #Scalar
    scale = np.array(K_X.shape[0]**-1).astype(K_X.dtype)
    if averaged_indices == (False, False):
        warn("No averaging correction computed. Returning None")
        return None
    if averaged_indices == (True, False):
        def _matvec(w):
            w = w.squeeze()
            return np.full_like(w, -np.dot(w, K_Y)*scale)
    elif averaged_indices == (False, True):
        def _matvec(w):
            w = w.squeeze()
            W = w.sum()
            return -K_X*W*scale
    else:
        def _matvec(w):
            w = w.squeeze()
            W = w.sum()
            #Default choice average both indices
            return (-K_X*W - np.dot(w, K_Y) + W*K_D*scale)*scale
    return LinearOperator((X.shape[0], X.shape[0]), matvec =  _matvec, dtype= K_X.dtype)
