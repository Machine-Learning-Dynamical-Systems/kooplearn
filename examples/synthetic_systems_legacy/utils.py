import numpy as np
import scipy.stats
import scipy.special
import h5py
import os
import datetime
from tqdm import tqdm
import scipy.integrate
from scipy.stats.sampling import NumericalInversePolynomial
import matplotlib.pyplot as plt
from pykeops.numpy import Pm
import sys
sys.path.append("../../")
from kooplearn.kernels import Kernel
from kooplearn.estimators import KernelRidge

class CosineKernel(Kernel):
    def __init__(self, N):
        self.N = N
        self.C_N = np.pi/scipy.special.beta(N//2  + 0.5, 0.5)
    
    def _preprocess(self, X):
        if np.ndim(X) == 1:
            X = X[:,None]
        elif np.ndim(X) == 0:
            X = np.array(X)[None,None]
        return X
    
    def __call__(self, X, Y=None, backend='auto'):
        X = self._preprocess(X)
        if Y is None:
            Y = X.copy()
        else:
            Y = self._preprocess(Y)

        if backend == 'keops':        
            return Pm(self.C_N)*(((np.pi*self.cdist(X,Y)).cos())**(Pm(self.N)))
        else:
            res = X - Y.T
            return self.C_N*((np.cos(np.pi * res))**self.N)

class CosineDistribution():
    def __init__(self, N):
        self.N = N
        self.C_N = np.pi/scipy.special.beta(N//2  + 0.5, 0.5)
    def pdf(self, x):
        return self.C_N*((np.cos(np.pi * x))**self.N)

class LogisticMap():
    def __init__(self, N=None):
        self._noisy = False      
        if N is not None:
            #Noisy case
            self._noisy = True
            self.N = N
            self.C_N = np.pi/scipy.special.beta(N//2  + 0.5, 0.5)
            self._evals, self._PF_largest_evec, self._Koop_evecs = self._transfer_matrix_eig_process()
            self._urng = np.random.default_rng()
            self._rng = NumericalInversePolynomial(self, domain=(0,1), random_state=self._urng)
            self._noise_dist = CosineDistribution(N)
            self._noise_rng = NumericalInversePolynomial(self._noise_dist, domain=(-0.5,0.5), mode = 0, random_state=self._urng)
        else:
            #Noiseless case
            pass

    def pdf(self, x):
        if self._noisy:
            if np.isscalar(x):
                y = 0
            else:
                y = np.zeros(x.shape)
            for i in range(self.N + 1):
                y += self._feature(x, i)*self._PF_largest_evec[i]
            return np.abs(y)
        else:
            return scipy.stats.beta(0.5, 0.5).pdf(x)

    def rvs(self, size=1):
        if np.isscalar(size):
            size = (size, 1)
        if self._noisy:
            return self._rng.rvs(size)
        else:
            return scipy.stats.beta(0.5, 0.5).rvs(size=size)
    
    def noise(self, size = 1):
        if np.isscalar(size):
            size = (size, 1)
        if self._noisy:
            return self._noise_rng.rvs(size)
        else:
            raise ValueError("This method not needed for noiseless case") 

    def _transfer_matrix(self):
        if self._noisy:
            N = self.N
            eps = 1e-10
            A = np.zeros((N + 1, N + 1))
            for i in tqdm(range(N + 1), desc='Init: Transfer matrix'):
                for j in range(N + 1):     
                    alpha = lambda x: self._feature(self.map(x), i)
                    beta = lambda x: self._feature(x, j)
                    f = lambda x: alpha(x)*beta(x)     
                    q = scipy.integrate.quad(f, 0, 1, epsabs=eps, epsrel=eps)
                    A[i, j] = q[0]
            return A
        else:
            raise ValueError("This method not needed for noiseless case")
    
    def _transfer_matrix_eig_process(self):
        if self._noisy:
            A = self._transfer_matrix()
            self._A = A
            ev, lv, rv = scipy.linalg.eig(A, left=True, right=True)
            invariant_eig_idx = None
            for idx, v in enumerate(ev):
                if np.isreal(v):
                    if np.abs(v - 1) < 1e-10:
                        invariant_eig_idx = idx
                        break
            if invariant_eig_idx is None:
                raise ValueError("No invariant eigenvalue found")
            PF_largest_evec = rv[:, invariant_eig_idx]
            if not np.all(np.isreal(PF_largest_evec)):
                print(f"Largest eigenvector is not real, largest absolute imaginary part is {np.abs(np.imag(PF_largest_evec)).max()}. Forcing it to be real.")
            return ev, np.real(PF_largest_evec), lv

        else:
            raise ValueError("This method not needed for noiseless case")
    
    def _feature(self, x, i):
        if self._noisy:
            N = self.N
            C_N = self.C_N
            return ((np.sin(np.pi * x))**(N - i))*((np.cos(np.pi * x))**i)*np.sqrt(scipy.special.binom(N, i)*C_N)
        else:
            raise ValueError("This method not needed for noiseless case")

    def map(self, x, noisy=False):
        if noisy:
            y = 4*x*(1 - x)
            if np.isscalar(x):
                xi = self.noise(1)[0]
            else:
                xi = self.noise(x.shape)
            return np.mod(y + xi, 1)
        else:
            return 4*x*(1 - x)


"""
statistics = {
    'num_test_samples' :    #Numer of test samples,
    'test_repetitions' :    #(Optional) Number of repetitions of the test,
    'train_repetitions' :   #(Optional) Number of repetitions of the training,
}

parameters = {
    'num_train_samples' :   #Number of training samples or array of training samples,
    'ranks' :               #Rank of the reduced rank approximation or array of ranks,
    'tikhonov_regs':        #Tikhonov regularization parameter or array of regularization parameters,
    'estimators' :          #Estimator or array of estimators,
}
"""
class LogisticMapSimulation:
    def __init__(self, kernel, parameters, statistics, N = None, iid=True):
        self.kernel = kernel
        self.parameters = parameters
        self.logistic = LogisticMap(N)
        self.iid = iid
        self.test_repetitions = statistics.get('test_repetitions', 1)
        self.train_repetitions = statistics.get('train_repetitions', 1)
        test_size = (statistics['num_test_samples'], self.test_repetitions)
        self.test_set = self._generate_pts(test_size)
        self._parameter_values = ['estimators', 'num_train_samples', 'ranks', 'tikhonov_regs']
        self._is_computed = False
    
    def _unpack_parameters(self):
        iterators = []
        for p_name in self._parameter_values:
            p = self.parameters[p_name]
            if np.ndim(p) == 0:
                p = np.array([p])
            iterators.append(np.array(p))
        return iterators
    
    def _preprocess_eigs(self, eigs):
        if np.all(np.isreal(eigs)):
            eigs = np.real(eigs)
        _perm = np.argsort(np.abs(eigs))
        eigs = eigs[_perm][::-1]
        return eigs

    def run_eigs(self, backend= 'auto', save = True, num = None, save_path = None):
        if save:
            if save_path is None:
                save_path = './'
            else:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            #Create file name based on time
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "data_" + time_str + ".h5"
            h5file = h5py.File(save_path + filename, "w")

        if self.logistic._noisy:
            iterators = self._unpack_parameters()
            estimators, num_train_samples, ranks, tikhonov_regs = iterators
            if num_train_samples.shape[0] > 1 or ranks.shape[0] > 1:
                    raise ValueError("Cannot return eigs if ranks or num_train_samples is an array")
            rank = ranks[0]
            num_train = num_train_samples[0]
            if num is None:
                eig_size = np.maximum(rank, num_train)
            else:
                eig_size = np.maximum(rank, num)

            #Mean, Std
            eig_array = 1e2*np.ones((self.train_repetitions, eig_size, estimators.shape[0], tikhonov_regs.shape[0]), dtype = np.complex) #Large init value so that when min distance to closest eig is computed avoid problems.
            train_errors = np.zeros((2, estimators.shape[0], tikhonov_regs.shape[0]))
            test_errors = np.zeros((2, estimators.shape[0], tikhonov_regs.shape[0]))

            if save:
                eig_dset = h5file.create_dataset("eigenvalues", data = eig_array)
                train_dset = h5file.create_dataset("train_errors", data = train_errors)
                test_dset = h5file.create_dataset("test_errors", data = test_errors)

            test_X, test_Y = self.test_set
            #Iterate over possible estimators
            for est_i, estimator in enumerate(estimators):
                X_train, Y_train = self._generate_pts((num_train, self.train_repetitions))          
                for reg_i, reg in enumerate(tikhonov_regs):
                    print(estimator.__name__ + f" regularization {reg_i + 1}/{tikhonov_regs.shape[0]}")
                    _err = np.zeros((self.train_repetitions, self.test_repetitions))
                    _tr_err = np.zeros(self.train_repetitions)
                    #Iterate over train repetitions
                    for train_rep in range(self.train_repetitions):
                        X, Y = X_train[:,train_rep][:,None], Y_train[:,train_rep][:,None]
                        if estimator == KernelRidgeRegression:
                            model = estimator(kernel = self.kernel, tikhonov_reg = reg)
                            model.fit(X, Y, backend = 'cpu')
                        elif estimator == PrincipalComponent:
                            model = estimator(kernel = self.kernel, rank = rank)
                            model.fit(X, Y, backend = backend)
                        else:
                            model = estimator(kernel = self.kernel, rank = rank, tikhonov_reg = reg)
                            model.fit(X, Y, backend = backend)
                        _fitted_evals = model.eigvals(k = num)
                        eig_array[train_rep, :, est_i, reg_i][np.arange(_fitted_evals.shape[0])] = _fitted_evals
                        _err[train_rep] = np.array([model.risk(test_X[:,k][:, None], test_Y[:,k][:, None]) for k in range(test_X.shape[1])])
                        _tr_err[train_rep] = model.risk()
                        if save:
                            eig_dset[...] = eig_array
                        
                    test_errors[0, est_i, reg_i] = np.mean(_err)
                    test_errors[1, est_i, reg_i] = np.std(_err)
                    
                    train_errors[0, est_i, reg_i] = np.mean(_tr_err)
                    train_errors[1, est_i, reg_i] = np.std(_tr_err)
                    if save:
                        train_dset[...] = train_errors
                        test_dset[...] = test_errors
            if save:
                h5file.close()
            return train_errors, test_errors, eig_array
        else:
            raise ValueError("Eigenvalue error not computable for noiseless case.")

    def run(self, backend = 'auto'):
        iterators = self._unpack_parameters()
        estimators, num_train_samples, ranks, tikhonov_regs = iterators
        size_err = (2,) #Mean, std  
        for i in iterators:
            size_err += i.shape
        errors = np.zeros(size_err)

        test_X, test_Y = self.test_set
        #Iterate over possible estimators
        for est_i, estimator in enumerate(estimators):
            _first_iter_found = False
            #Iterate over possible training sizes
            _train_iter, _first_iter_found = self._dress_iterator(num_train_samples, estimator.__name__, _first_iter_found)  
            for train_i, num_train in _train_iter:
                X_train, Y_train = self._generate_pts((num_train, self.train_repetitions))
                #Iterate over possible ranks
                _rank_iter, _first_iter_found = self._dress_iterator(ranks, estimator.__name__, _first_iter_found)
                for rank_i, rank in _rank_iter:
                    #Iterate over possible regularization parameters
                    _reg_iter, _first_iter_found = self._dress_iterator(tikhonov_regs, estimator.__name__, _first_iter_found)
                    for reg_i, reg in _reg_iter:
                        _err = np.zeros((self.train_repetitions, self.test_repetitions))
                        #Iterate over train repetitions
                        for train_rep in range(self.train_repetitions):
                            X, Y = X_train[:,train_rep][:,None], Y_train[:,train_rep][:,None]
                            if estimator == KernelRidgeRegression:
                                model = estimator(kernel = self.kernel, tikhonov_reg = reg)
                            else:
                                model = estimator(kernel = self.kernel, rank = rank, tikhonov_reg = reg)
                            model.fit(X, Y, backend = backend)
                            _err[train_rep] = np.array([model.risk(test_X[:,k][:, None], test_Y[:,k][:, None]) for k in range(test_X.shape[1])])
                        errors[0, est_i, train_i, rank_i, reg_i] = np.mean(_err)
                        errors[1, est_i, train_i, rank_i, reg_i] = np.std(_err)
        self._is_computed = True
        self._errors = errors
        return iterators, errors
    
    def _dress_iterator(self, iterator, desc, found):
        _iter = enumerate(iterator)
        if (not found) and (iterator.shape[0] > 1):
                found = True
                _iter = tqdm(_iter, desc = desc, total = iterator.shape[0]) 
        return _iter, found

    def _generate_pts(self, size):
        if np.isscalar(size):
            size = (size, 1)
        if self.iid:
            X = self.logistic.rvs(size)
            Y = self.logistic.map(X, noisy = self.logistic._noisy)       
        else:
            _raw = np.zeros((size[0] + 1, size[1]))
            _raw[0] = self.logistic.rvs(size[1])
            for i in range(1, size[0] + 1):
                _raw[i] = self.logistic.map(_raw[i - 1], noisy = self.logistic._noisy)
            X = _raw[:-1]
            Y = _raw[1:]
        return X, Y

    def plot1D(self, variables_idx_dict, style= '-' ):
        if not self._is_computed:
            self.run()
        
        for var in self._parameter_values[1:]:
            variables_idx_dict[var] = variables_idx_dict.get(var, 0)
            if variables_idx_dict[var] == None:
                variables_idx_dict[var] = slice(None)
                _var_name = var     
        errors = self._errors[:,:,variables_idx_dict['num_train_samples'], variables_idx_dict['ranks'], variables_idx_dict['tikhonov_regs']]
        iterators = self._unpack_parameters()
        fig, ax = plt.subplots()
        for idx, estimator in enumerate(iterators[0]):       
            mean = errors[0,idx]
            std = errors[1,idx]
            ax.fill_between(self.parameters[_var_name], mean - std, mean + std, alpha = 0.1)
            ax.plot(self.parameters[_var_name], mean, style, label = estimator.__name__)
            ax.margins(0)
            ax.legend(frameon=False)
            ax.set_ylabel("Test error")        
        return fig, ax