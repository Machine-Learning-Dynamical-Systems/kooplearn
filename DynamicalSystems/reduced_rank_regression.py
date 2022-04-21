
from cmath import sqrt
from genericpath import exists
from matplotlib.pyplot import axis
from scipy.sparse.linalg import eigs, eigsh, aslinearoperator, LinearOperator, cg
from scipy.linalg import eig, eigh, solve,qr
from scipy.sparse import diags
from pykeops.numpy import Vi
from DynamicalSystems.utils import modified_QR, parse_backend, _check_real, IterInv
from DynamicalSystems.kernels import Linear
import numpy as np
from warnings import warn

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

class KoopmanRegression:
    def __init__(self, kernel = None, regularizer = None, rank=None, center_kernel=False, backend='auto', rank_reduction='r3', random_state = 0, powers = 2, offset = None, tol = 1e-12):
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = Linear()
        self.rank = rank
        self.reg = regularizer
        self.center_kernel = center_kernel
        self.rank_reduction = rank_reduction
        self.n_samples = None
        self.random_state = random_state
        self.powers = powers 
        self.offset = offset
        self.tol = tol
        self.backend = backend
        self.modes_ = None
        self.evals = None
        self.revecs = None
        self.levecs = None
              

    def fit(self, data, evolved_data):
        self.X, self.Y = data, evolved_data
        self.backend = parse_backend(self.backend, data)
        self.n_samples = self.X.shape[0]

        if self.backend == 'keops':
            self.K_X = aslinearoperator(self.kernel(self.X, backend=self.backend))
            self.K_Y = aslinearoperator(self.kernel(self.Y, backend=self.backend))
            self.K_YX = aslinearoperator(self.kernel(self.Y, self.X, backend=self.backend))
        else:
            self.K_X = self.kernel(self.X, backend=self.backend)
            self.K_Y = self.kernel(self.Y, backend=self.backend)
            self.K_YX = self.kernel(self.Y, self.X, backend=self.backend)
        
        self.dtype = self.K_X.dtype
        
        if self.center_kernel:
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
            if self.rank_reduction == 'r3':
                #print("INIT: Computing low-rank-projection via Reduced Rank Regression")
                self.V, self.U = self.r3()
            elif self.rank_reduction == 'pcr':
                #print("INIT: Computing low-rank-projection via Principal Component Regression")
                self.V, self.U = self.pcr()
            elif self.rank_reduction == 'r4':
                #print("INIT: Computing low-rank-projection via Randomized Reduced Rank Regression")
                self.V, self.U = self.r4()
            else:
                raise ValueError(f"Unrecognized backend '{self.rank_reduction}'. Accepted values are None, 'r3', 'r4' and 'pcr'.")
        else:
            pass

        self.fit_eig()
        self.fit_modes()
        self.fit_dmd()
        self.fit_forcast()

    def test(self, data = None, error = 'risk'):
        if error == 'risk':
            return self.risk(data)
        elif error == 'spectral':
            return self.spectral_error(data)
        else:
            raise ValueError(f"Unrecognized error '{error}'. Accepted values are 'risk' and 'spectral'.")

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
        
    def r3(self):
        #For the moment data_kernel = LinearOperator, evolved_data_kernel = LinearOperator
        dim = self.n_samples
        inverse_dim = self.n_samples**-1        
        
        if self.reg is not None:
            alpha =  self.reg*dim 
            K = inverse_dim*(self.K_Y@self.K_X)
            if self.backend == 'keops':
                tikhonov = aslinearoperator(diags(np.ones(dim, dtype=self.dtype)*alpha))
                Minv = IterInv(self.kernel, self.X, alpha)
                sigma_sq, U = eigs(K, self.rank, self.K_X + tikhonov,  Minv=Minv)                
            else:
                tikhonov = np.eye(dim, dtype=self.dtype)*(self.reg*dim)   
                sigma_sq, U = eig(K, self.K_X + tikhonov)
        
            assert np.max(np.abs(np.imag(sigma_sq))) < self.tol
            
            sigma_sq = np.real(sigma_sq)
            sort_perm = np.argsort(sigma_sq)[::-1]
            sigma_sq = sigma_sq[sort_perm][:self.rank]
            U = U[:,sort_perm][:,:self.rank]

            # #Check that the eigenvectors are real (or have a global phase at most)
            # if not _check_real(U):
            #     U_global_phase_norm = np.angle(U).std(axis=0)
            #     if U_global_phase_norm.mean()  > 1e-8:
            #         raise ValueError("Computed projector is not real. The kernel function is either severely ill conditioned or non-symmetric")
            #     else:
            #         #It has a global complex phase, take absolute.
            #         U = np.real(U@np.diag(U_global_phase_norm)) ## WRONG
            # else:
            U = np.real(U)
            
            U, R = modified_QR(U, self.backend, inverse_dim*(self.K_X@(self.K_X + tikhonov)))
            if np.linalg.norm(R-np.diag(np.diag(R)), ord = 1)>self.tol:
                print('orthogonality error:', np.linalg.norm(R-np.diag(np.diag(R)), ord = 1))
            #print(inverse_dim*U.T@(self.K_X@(self.K_X + tikhonov)@U))
            V = (self.K_X@np.asfortranarray(U))            
        else:
            if self.backend == 'keops':
                raise ValueError(f"Unsupported backend '{self.backend}' without the regularization.")
            else:
                sigma_sq, V = eigsh(self.K_Y, self.rank)

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
    
    def fit_eig(self, X=None):
        dim = self.K_X.shape[0]
        dim_inv = dim**(-1)

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
            if self.reg is not None:
                tikhonov = np.eye(dim, dtype=self.K_X.dtype)*(self.reg*dim)
                C = solve(self.K_X+tikhonov, self.K_YX, assume_a='pos') 
                vals, lv, rv =  eig(C, left=True, right=True)
                self.evals = vals 
                self.revecs = rv
                self.levecs = lv
            else:
                #C = solve(self.K_X, self.K_YX, assume_a='sym')
                C = np.linalg.pinv(self.K_X)@ self.K_YX
                vals, lv, rv =  eig(C, left=True, right=True)
                self.evals = vals 
                self.revecs = rv
                self.levecs = lv

        normalize_ = self.evals.shape[0]
        self.revecs = self.revecs @ np.diag(1/np.sqrt(normalize_*np.diag(self.revecs.T@(self.K_X@self.revecs))))
        self.levecs = self.levecs @ np.diag(1/np.sqrt(normalize_*np.diag(self.levecs.T@(self.K_Y@self.levecs))))

        self.refuns = lambda x: np.sqrt(dim_inv)*self.kernel(x[np.newaxis],self.X,backend=self.backend) @ self.revecs
        self.lefuns = lambda x: np.sqrt(dim_inv)*self.kernel(x[np.newaxis],self.Y,backend=self.backend) @ self.levecs

        return self.evals, self.levecs, self.revecs
    
    def fit_modes(self, f = None):
        if f is not None:
            observable = f(self.X)
            if len(observable.shape)==1:
                observable = observable[:,np.newaxis]
        else:
            observable = self.X   
        U_tilde = np.matrix(self.revecs)
        if self.backend == 'keops':
            F = U_tilde.H.matmat(np.asfortranarray(observable))
            D = (U_tilde.H)@self.K_X.matmat(np.asfortranarray(self.revecs))
        else:
            F = U_tilde.H@observable
            D = (U_tilde.H)@self.K_X@U_tilde
        self.modes = np.sqrt(self.n_samples)*solve(D, F, assume_a='her')    
        return self.modes

    def fit_forcast(self, f = None):
        if f is not None:
            observable_X = f(self.X)
            observable_Y = f(self.Y)
            # if len(observable_X.shape)==1:
            #     observable_X = observable_X[:,np.newaxis]
            # if len(observable_Y.shape)==1:
            #     observable_Y = observable_Y[:,np.newaxis]
        else:
            observable_X = self.X
            observable_Y = self.Y     
        
        if self.rank is not None:
            #self.backcast = lambda x: observable_X.T @ self.U @ self.V.T @ self.kernel(self.Y, x, backend=self.backend) / self.n_samples
            self.backcast = lambda x: self.kernel(x, self.Y, backend=self.backend) @ self.V @self.U.T @ observable_X  / self.n_samples
            #self.forcast = lambda x: observable_Y.T @ self.V @ self.U.T @ self.kernel(self.X, x, backend=self.backend) / self.n_samples
            self.forcast = lambda x: self.kernel(x, self.X, backend=self.backend) @ self.U @self.V.T @ observable_Y  / self.n_samples
        else:
            if self.reg is not None:
                tikhonov = np.eye(self.n_samples, dtype=self.dtype)*(self.reg*self.n_samples)
                self.backcast = lambda x:  solve( self.K_X + tikhonov, self.kernel(self.Y, x, backend=self.backend), assume_a='pos').T @ observable_X
                self.forcast = lambda x:  solve( self.K_X + tikhonov, self.kernel(self.X, x, backend=self.backend), assume_a='pos').T @ observable_Y
            else:
                self.backcast = lambda x: solve( self.K_X, self.kernel(self.Y, x, backend=self.backend), assume_a='sym').T @ observable_X
                self.forcast = lambda x: solve( self.K_X, self.kernel(self.X, x, backend=self.backend), assume_a='sym').T @ observable_Y
        


    # def fit_forecast(self):
    #     if self.rank == None:
    #         dim = self.K_X.shape[0]
    #         dt = self.K_X.dtype
    #         if self.backend == 'keops':     
    #             alpha = self.reg*dim
    #             M = self.kernel(self.X, backend=self.backend)
    #             _X = Vi(self.Y)
    #             self._forecast_Z =M.solve(_X, alpha = alpha, eps = 1e-6)
    #         else:
    #             tikhonov = np.eye(dim, dtype=dt)*(self.reg*dim)
    #             K_reg = self.K_X + tikhonov
    #             self._forecast_Z = solve(K_reg,self.Y, assume_a='pos') 
    #     else:
    #             self._forecast_Z = (self.V.T)@self.Y

    # def forecast(self, initial_point):
    #     try:
    #         if self.rank == None:
    #             _forecast_S = self.kernel(initial_point[np.newaxis, :], self.X, backend = 'cpu')
    #         else:
    #             _init_k = self.kernel(initial_point[np.newaxis, :], self.X, backend = 'cpu')
    #             if self.backend == 'keops':
    #                 _forecast_S = aslinearoperator(_init_k).matmat(self.U)
    #             else:
    #                 _forecast_S = _init_k@self.U
    #         return np.squeeze(_forecast_S@self._forecast_Z)
    #     except AttributeError:
    #         self.fit_forecast()
    #         return self.forecast(initial_point)

    def fit_dmd(self, f = None, time_step = 1, which = None):
        
        dim = self.X.shape[1]
        n = self.X.shape[0]

        if f is not None:
            self.fit_modes(f)

        if which is not None:
            evals = self.evals[which]
            revecs = self.revecs[:,which]
            if len(self.modes.shape) == 1:
                modes = self.modes[which]                
            else:
                modes = self.modes[which,:]
        else:
            evals = self.evals
            revecs = self.revecs
            modes = self.modes
        omegas = np.log(evals)/time_step
        dmd = lambda x0, t: np.sqrt(1/self.n_samples)*np.real((np.exp(t[np.newaxis].T @ omegas[np.newaxis]) @ np.diag((self.kernel(x0[np.newaxis],self.X,backend=self.backend) @ revecs)[0]) ) @ modes)
        self.dmd = dmd
        return dmd


    def spectral_error(self, data = None):
        if data is None:
            KX = self.K_X
            KYX = self.K_YX
            normalize_ = self.n_samples
        else:
            normalize_ = np.sqrt(self.n_samples*data[0].shape[0]) 
            KX = self.kernel(data[0], self.X, backend=self.backend)
            KYX = self.kernel(data[1], self.X, backend=self.backend)

        return np.linalg.norm(KYX @ self.revecs - KX @ self.revecs@np.diag(self.evals), ord = 'fro') /normalize_

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
                if self.reg is None:
                    C = solve(self.K_X, self.K_X ,assume_a='sym' ) 
                    risk -= np.trace(C @ self.K_Y)
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
            risk = np.trace(self.kernel(data[1], data[1], backend=self.backend)) / data[0].shape[0]
            if self.rank is None:
                if self.reg is None:
                    C = solve(self.K_X, self.kernel(self.X, data[0], backend=self.backend),assume_a='sym' ) * self.n_samples / normalize_
                    risk -= 2*np.trace( C @ self.kernel(data[1], self.Y, backend=self.backend)) / normalize_
                    risk += np.trace( C.T @ self.K_Y @ C)  / self.n_samples
                else:
                    tikhonov = np.eye(self.n_samples, dtype=self.dtype)*(self.reg*self.n_samples) 
                    C = solve(self.K_X+tikhonov, self.kernel(self.X, data[0], backend=self.backend),assume_a='sym' ) * self.n_samples / normalize_
                    risk -= 2*np.trace( C @ self.kernel(data[1], self.Y, backend=self.backend)) / normalize_
                    risk += np.trace( C.T @ self.K_Y @ C)  / self.n_samples

            else:
                U1 = (self.kernel(data[0],self.X, backend=self.backend) @ self.U) / normalize_
                V1 = (self.kernel(data[1], self.Y, backend=self.backend)  @ self.V) / normalize_
                C = self.V.T @ (self.K_Y @ self.V) / self.n_samples
                risk -= 2*np.trace(V1.T@U1)
                risk += np.trace((U1.T@U1) @ C)

        return risk

    