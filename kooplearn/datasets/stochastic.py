from typing import Optional
from kooplearn.datasets.misc import DataGenerator, DiscreteTimeDynamics, LinalgDecomposition
from kooplearn._src.utils import topk
import numpy as np
import logging
import scipy
import scipy.sparse
from scipy.integrate import quad, romb
from scipy.stats import beta
from scipy.special import binom
from scipy.stats.sampling import NumericalInversePolynomial
import math
from pathlib import Path
logger = logging.getLogger('kooplearn')

try:
    import sdeint
    _has_sdeint = True
except ImportError:
    _has_sdeint = False

class MockData(DataGenerator):
    def __init__(self, num_features: int = 50, rng_seed: Optional[int] = None):
        self.rng = np.random.default_rng(rng_seed)
        self.num_features = num_features

    def sample(self, X0: np.ndarray, T: int = 1):
        return self.rng.random((T + 1, self.num_features))

class LinearModel(DiscreteTimeDynamics):
    def __init__(self, A: np.ndarray, noise: float = 0., rng_seed: Optional[int] = None):
        self.A = A
        self.noise = noise
        self.rng = np.random.default_rng(rng_seed)

    def _step(self, X: np.ndarray):
        return self.A @ X + self.noise * self.rng.standard_normal(size=X.shape)

class RegimeChangeVAR(DiscreteTimeDynamics):
    def __init__(self, phi1: np.ndarray, phi2: np.ndarray, transition: np.ndarray, noise: float = 0.,
                 rng_seed: Optional[int] = None):
        self.phi1 = phi1
        self.phi2 = phi2
        self.transition = transition
        self.noise = noise
        self.rng = np.random.default_rng(rng_seed)
        self.current_state = 0

    def _step(self, X: np.ndarray):
        rand_trans = np.random.uniform(0, 1)
        if rand_trans < self.transition[self.current_state, 0]:
            self.current_state = 0
            return self.phi1 @ X + self.noise * self.rng.standard_normal(size=X.shape)
        else:
            self.current_state = 1
            return self.phi2 @ X + self.noise * self.rng.standard_normal(size=X.shape)

# Noisy Logistic Map
class CosineDistribution:
    def __init__(self, N):
        self.N = N
        self.C_N = np.pi / scipy.special.beta(N // 2 + 0.5, 0.5)

    def pdf(self, x):
        return self.C_N * ((np.cos(np.pi * x)) ** self.N)


class LogisticMap(DiscreteTimeDynamics):
    def __init__(self, r: float = 4.0, N: Optional[int] = None, rng_seed: Optional[int] = None):
        self.rng_seed = rng_seed
        self.r = r
        self._rng = np.random.default_rng(self.rng_seed)
        if N is not None:
        #Noisy logistic map
            if N % 2 != 0:
                raise ValueError("N must be even")
            self.has_noise = True
            self.N = N
            self._noise_rng = NumericalInversePolynomial(
                CosineDistribution(N), 
                domain=(-0.5, 0.5), 
                mode=0, 
                random_state=self._rng)
            self._svdvals, self._eigvals, self._lv, self._rv = self._init_transfer_matrices()            
        
        else:
            #Noiseless logistic map
            self.has_noise = False

    def predict(self, X_init: np.ndarray, T: int = 1):
        raise NotImplementedError("This method is not implemented yet")

    def eig(self, eval_left_on: Optional[np.ndarray] = None, eval_right_on: Optional[np.ndarray] = None):
        if not self.has_noise:
            raise ValueError("This method is only available for the noisy logistic map")
        perron_eig_idx = np.argmax(np.abs(self._eigvals))

        if eval_left_on is None and eval_right_on is None: #None
            return self._eigvals
        elif eval_left_on is not None and eval_right_on is None: #Only left
            Xl = np.asanyarray(eval_left_on).reshape(-1)
            betas_mat = np.stack([self.noise_feature(Xl, i) for i in range(self.N + 1)], axis=1) # [Xl.shape[0], N + 1]
            
            lfuncs = self._eigvals, betas_mat @ self._lv
            #Standardize sign
            perron_fun_sign = np.sign(np.sign(lfuncs[:, perron_eig_idx]).mean())
            return lfuncs*perron_fun_sign
        elif eval_left_on is None and eval_right_on is not None: #Only right
            Xr = np.asanyarray(eval_right_on).reshape(-1)
            alphas_mat = np.stack([self.noise_feature_composed_map(Xr, i) for i in range(self.N + 1)], axis=1) # [Xr.shape[0], N + 1]
            rfuncs = self._eigvals, alphas_mat @ self._rv
            perron_fun_sign = np.sign(np.sign(rfuncs[:, perron_eig_idx]).mean())
            return rfuncs*perron_fun_sign
        
        else: #All
            #Left
            Xl = np.asanyarray(eval_left_on).reshape(-1)
            betas_mat = np.stack([self.noise_feature(Xl, i) for i in range(self.N + 1)], axis=1) # [Xl.shape[0], N + 1]  
            lfuncs = betas_mat @ self._lv
            #Right
            Xr = np.asanyarray(eval_right_on).reshape(-1)
            alphas_mat = np.stack([self.noise_feature_composed_map(Xr, i) for i in range(self.N + 1)], axis=1) # [Xr.shape[0], N + 1]
            rfuncs = alphas_mat @ self._rv
            perron_fun_sign = np.sign(np.sign(rfuncs[:, perron_eig_idx]).mean())
            return self._eigvals, lfuncs*perron_fun_sign, rfuncs*perron_fun_sign
        
    def svals(self):
        if not self.has_noise:
            raise ValueError("This method is only available for the noisy logistic map")  
        return self._svdvals
    
    def _step(self, X_0: np.ndarray):
        return self.map(X_0)

    def map(self, X_init: np.ndarray, noisy: Optional[bool] = None):
        if noisy is None:
            noisy = self.has_noise
        #Dim checks on X_init
        if X_init.ndim == 0:
            X_init = X_init.reshape(1, 1)
        elif X_init.ndim == 1:
            X_init = X_init.reshape(-1, 1)
        elif X_init.ndim > 2:
            raise ValueError("X_init must be a scalar, a 1D array or a 2D array")
        if noisy:
            y = self.r * X_init * (1 - X_init)
            assert hasattr(self, "_noise_rng"), "Noise rng not initialized, initialize the logistic map with N != None"
            xi = self._noise_rng.rvs(X_init.shape)
            return np.mod(y + xi, 1)
        else:
            return self.r * X_init * (1 - X_init)
    
    def noise_feature(self, x, i): #beta
        if not self.has_noise:
            raise ValueError("This method is only available for the noisy logistic map")   
        N = self.N
        normalization_cst = np.pi / scipy.special.beta(N // 2 + 0.5, 0.5)
        return ((np.sin(np.pi * x)) ** (N - i)) * ((np.cos(np.pi * x)) ** i) * np.sqrt(binom(N, i) * normalization_cst)

    def noise_feature_composed_map(self, x, i): #alpha = beta circ logistic
        r = self.r
        return self.noise_feature(r * x * (1 - x), i)

    def _init_transfer_matrices(self):
        if not self.has_noise:
            raise ValueError("This method is only available for the noisy logistic map")   
        N = self.N

        def alphas_mat_el(i: int, j: int):
            def pairing(x):
                return self.noise_feature_composed_map(x, i) * self.noise_feature_composed_map(x, j)
            return scipy.integrate.quad(pairing, 0, 1)[0]

        def betas_mat_el(i: int, j: int):
            def pairing(x):
                return self.noise_feature(x, i) * self.noise_feature(x, j)
            return scipy.integrate.quad(pairing, 0, 1)[0]

        def koopman_el(i: int, j: int):
            def pairing(x):
                return self.noise_feature(x, i) * self.noise_feature_composed_map(x, j)
            return scipy.integrate.quad(pairing, 0, 1)[0]

        K = np.array([[koopman_el(i, j) for j in range(N + 1)] for i in range(N + 1)])
        A = np.array([[alphas_mat_el(i, j) for j in range(N + 1)] for i in range(N + 1)])
        B = np.array([[betas_mat_el(i, j) for j in range(N + 1)] for i in range(N + 1)])

        eigvals, lv, rv = scipy.linalg.eig(K, left=True, right=True)
        svdvals = scipy.linalg.eigvals(B@A)
        #Check that svdvals are real and positive
        if not np.all(np.isreal(svdvals)):
            logger.warning("Singular values are not real, taking the real part")
        svdvals = svdvals.real
        
        if min(svdvals) < -np.finfo(svdvals.dtype).eps*len(svdvals): 
            logger.warning("Singular values are not positive, truncating to positive values")
            svdvals = np.where(svdvals < 0, 0, svdvals)

        return svdvals, eigvals, lv, rv
                                          

class _legacyLogisticMap(DiscreteTimeDynamics):
    def __init__(self, r: float = 4.0, N: Optional[int] = None, rng_seed: Optional[int] = None):
        self._noisy = False
        self.ndim = 1
        self.rng_seed = rng_seed
        self.r = r
        if N is not None:
            assert N % 2 == 0
            # Noisy case
            self._noisy = True
            self.N = N
            self.C_N = np.pi / scipy.special.beta(N // 2 + 0.5, 0.5)
            self._evals, self._PF_largest_evec, self._Koop_evecs = self._transfer_matrix_eig_process()
            self._urng = np.random.default_rng(rng_seed)
            self._rng = NumericalInversePolynomial(self, domain=(0, 1), random_state=self._urng)
            self._noise_dist = CosineDistribution(N)
            self._noise_rng = NumericalInversePolynomial(
                self._noise_dist, domain=(-0.5, 0.5), mode=0, random_state=self._urng)
        else:
            # Noiseless case
            pass

    def eig(self, num_right_fn_evals: Optional[int] = None):
        assert self._noisy, "Eigenvalue decomposition is only available for the noisy Logistic Map"
        if num_right_fn_evals is not None:
            _x = np.linspace(0, 1, num_right_fn_evals)
            _x = _x[:, None]

            phi_X = np.zeros((num_right_fn_evals, self.N + 1))
            for n in range(self.N + 1):
                phi_X[:, n] = self._feature(self.map(_x), n)[:, 0]

            ref_evd = LinalgDecomposition(
                self._evals,
                _x,
                phi_X @ self._Koop_evecs
            )
            return ref_evd
        else:
            return self._evals

    def _step(self, X_0: np.ndarray):
        return self.map(X_0, noisy=self._noisy)

    def pdf(self, x):
        if self._noisy:
            if np.isscalar(x):
                y = 0
            else:
                y = np.zeros(x.shape)
            for i in range(self.N + 1):
                y += self._feature(x, i) * self._PF_largest_evec[i]
            return np.abs(y)
        else:
            return beta(0.5, 0.5).pdf(x)

    def rvs(self, size=1):
        if np.isscalar(size):
            size = (size, 1)
        if self._noisy:
            return self._rng.rvs(size)
        else:
            return beta(0.5, 0.5).rvs(size=size)

    def noise(self, size=1):
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
            for i in range(N + 1):
                for j in range(N + 1):
                    alpha = lambda x: self._feature(self.map(x), i)
                    beta = lambda x: self._feature(x, j)
                    f = lambda x: alpha(x) * beta(x)
                    q = quad(f, 0, 1, epsabs=eps, epsrel=eps)
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
                print(
                    f"Largest eigenvector is not real, largest absolute imaginary part is "
                    f"{np.abs(np.imag(PF_largest_evec)).max()}. Forcing it to be real.")
            return ev, np.real(PF_largest_evec), lv

        else:
            raise ValueError("This method not needed for noiseless case")

    def _feature(self, x, i):
        if self._noisy:
            N = self.N
            C_N = self.C_N
            return ((np.sin(np.pi * x)) ** (N - i)) * ((np.cos(np.pi * x)) ** i) * np.sqrt(binom(N, i) * C_N)
        else:
            raise ValueError("This method not needed for noiseless case")

    def map(self, x, noisy=False):
        if noisy:
            y = self.r * x * (1 - x)
            if np.isscalar(x):
                xi = self.noise(1)[0]
            else:
                xi = self.noise(x.shape)
            return np.mod(y + xi, 1)
        else:
            return self.r * x * (1 - x)


class MullerBrownPotential(DataGenerator):
    def __init__(self, kt: float = 1.5e3, rng_seed: Optional[int] = None):
        if not _has_sdeint:
            raise ImportError("sdeint not found, please install it to use this class")
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0.0, 0.0, 11.0, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])
        self.A = np.array([-200, -100, -170, 15])
        self.X = np.array([1, 0, -0.5, -1])
        self.Y = np.array([0, 0.5, 1.5, 1])
        self.kt = kt
        self.rng = np.random.default_rng(rng_seed)

    def sample(self, X0: np.ndarray, T: int = 1):
        tspan = np.arange(0, 0.1 * T, 0.1)
        result = sdeint.itoint(self.neg_grad_potential, self.noise_term, X0, tspan, self.rng)
        return result

    def potential(self, X: np.ndarray):
        t1 = X[0] - self.X
        t2 = X[1] - self.Y
        tinexp = self.a * (t1 ** 2) + self.b * t1 * t2 + self.c * (t2 ** 2)
        return np.dot(self.A, np.exp(tinexp))

    def neg_grad_potential(self, x: np.ndarray, t: np.ndarray):
        """
        A_j * exp[a_j * (x1^2 - 2 * x1 * X_j) + b_j * (x1 * x2 - x1 * Y_j)]

        -> grad inner exp:
            a_j * (2 * x1 - 2 * X_j) + b_j * (x2 - Y_j)
        """
        t1 = x[0] - self.X
        t2 = x[1] - self.Y
        grad_inner_exp_x1 = 2 * self.a * t1 + self.b * t2
        grad_inner_exp_x2 = 2 * self.c * t2 + self.b * t1
        tinexp = np.exp(self.a * (t1 ** 2) + self.b * t1 * t2 + self.c * (t2 ** 2))

        return -np.array([
            np.dot(self.A, tinexp * grad_inner_exp_x1),
            np.dot(self.A, tinexp * grad_inner_exp_x2),
        ]) / self.kt

    def noise_term(self, x: np.ndarray, t: np.ndarray):
        return np.diag([math.sqrt(2 * 1e-2), math.sqrt(2 * 1e-2)])

class LangevinTripleWell1D(DiscreteTimeDynamics):
    def __init__(self, gamma: float = 0.1, kt: float = 1.0, dt: float = 1e-4, rng_seed: Optional[int] = None):
        self.gamma = gamma
        self.kt = kt
        self.rng = np.random.default_rng(rng_seed)
        self.dt = dt
        
        self._inv_gamma = (self.gamma) ** -1

    def eig(self):
        if not hasattr(self, "_ref_evd"):
            self._compute_ref_evd()
        return self._ref_evd

    def _compute_ref_evd(self):
        assets_path = Path(__file__).parent / "assets"

        lap_x = scipy.sparse.load_npz(assets_path / "1D_triple_well_lap_x.npz")
        grad_x = scipy.sparse.load_npz(assets_path / "1D_triple_well_grad_x.npz")
        cc = np.load(assets_path / "1D_triple_well_cc.npz")

        force = scipy.sparse.diags(self.force_fn(cc))

        # Eq.(31) of https://doi.org/10.1007/978-3-642-56589-2_9 recalling that \sigma^2/(\gamma*2) = kBT
        generator = self._inv_gamma*self.kt*lap_x + self._inv_gamma*force.dot(grad_x)    
        generator = generator*self.dt

        vals, vecs = np.linalg.eig(generator.toarray())

        #Filter out timescales smaller than dt
        mask = np.abs(vals.real) < 1.0
        vals = vals[mask]
        vecs = vecs[:, mask]
        
        vals = np.exp(vals)
        #Checking that the eigenvalues are real
        type_ = vals.dtype.type
        f = np.finfo(type_).eps
   
        tol = f * 1000
        if not np.all(np.abs(vals.imag) < tol):
            raise ValueError("The computed eigenvalues are not real, try to decrease dt")
        else:
            vals = vals.real
        
        _k = len(vals)
        evd_sorting_perm = topk(vals, _k)
        vals = evd_sorting_perm.values
        vecs = vecs[:, evd_sorting_perm.indices].real

        dx = cc[1] - cc[0]
        boltzmann_pdf = (dx**-1)*scipy.special.softmax(self.force_fn(cc)/self.kt)
        abs2_eigfun = (np.abs(vecs)**2).T
        eigfuns_norms = np.sqrt(romb(boltzmann_pdf*abs2_eigfun, dx = dx, axis = -1))
        vecs = vecs*(eigfuns_norms**-1.0)

        #Storing the results
        self._ref_evd = LinalgDecomposition(vals, cc, vecs)
        self._ref_boltzmann_density = boltzmann_pdf

    def _step(self, X: np.ndarray):
        F = self.force_fn(X)
        xi = self.rng.standard_normal(X.shape)
        dX = F * self._inv_gamma * self.dt + np.sqrt(2.0 * self.kt * self.dt * self._inv_gamma) * xi
        return X + dX

    def force_fn(self, x: np.ndarray):
        return -1. * (-128 * np.exp(-80 * ((-0.5 + x) ** 2)) * (-0.5 + x) - 512 * np.exp(-80 * (x ** 2)) * x + 32 * (
                x ** 7) - 160 * np.exp(-40 * ((0.5 + x) ** 2)) * (0.5 + x))

    def _eigfun_sign_phase(self, estimated, true):
        norm_p = np.linalg.norm(estimated + true)
        norm_m = np.linalg.norm(estimated - true)
        if norm_p <= norm_m:
            return -1.0
        else:
            return 1.0

    def _standardize_evd(self, evd: LinalgDecomposition, dx: float,
                         density: Optional[np.ndarray] = None) -> LinalgDecomposition:
        # Sorting and normalizing
        sort_perm = np.flip(np.argsort(evd.values.real))
        functions = (evd.vectors[:, sort_perm]).real
        abs2_eigfun = (np.abs(functions) ** 2).T
        if density is not None:
            abs2_eigfun *= density
        # Norms
        funcs_norm = np.sqrt(romb(abs2_eigfun, dx=dx, axis=-1))
        functions *= (funcs_norm ** -1.0)
        values = (evd.values.real)[sort_perm]
        return LinalgDecomposition(values, functions)

    def standardize_eigenfunction_phase(self, evd: LinalgDecomposition) -> LinalgDecomposition:
        assert np.allclose(evd.x, self._ref_domain_sample)

        ref_evd = self._ref_evd
        density = self._ref_boltzmann_density
        dx = self._ref_domain_sample[1] - self._ref_domain_sample[0]

        # ref_evd is assumed already standardized
        evd = self._standardize_evd(evd, dx, density=density)
        phase_aligned_funcs = evd.vectors.copy()
        num_funcs = evd.vectors.shape[1]
        for r in range(num_funcs):
            estimated = evd.vectors[:, r]
            true = ref_evd.vectors[:, r]
            phase_aligned_funcs[:, r] = self._eigfun_sign_phase(estimated * density, true * density) * estimated
        return LinalgDecomposition(evd.values, phase_aligned_funcs)
