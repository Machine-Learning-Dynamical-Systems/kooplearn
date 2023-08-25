import pytest
from scipy.stats import special_ortho_group
from shutil import rmtree
import numpy as np
from pathlib import Path
from kooplearn._src.models import KernelDMD
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern
from kooplearn.data.datasets.stochastic import LinearModel


TRUE_RANK = 5
DIM = 20
NUM_SAMPLES = 50

def make_linear_system():
    eigs = 9*np.logspace(-3, -1, TRUE_RANK)
    # print("Eigenvalues:")
    # for ev in eigs:
    #     print(f"{ev:.1e} \t", end='')
    eigs = np.concatenate([eigs, np.zeros(DIM-TRUE_RANK)])
    Q = special_ortho_group(DIM, 0).rvs(1)
    A = np.linalg.multi_dot([Q, np.diag(eigs), Q.T])

    #Consistency-check
    assert np.allclose(np.sort(np.linalg.eigvalsh(A)), np.sort(eigs))
    return LinearModel(A, noise=1e-5, rng_seed=0)


@pytest.mark.parametrize('kernel', [DotProduct(), RBF(), Matern()])
@pytest.mark.parametrize('reduced_rank', [True, False])
@pytest.mark.parametrize('rank', [TRUE_RANK, TRUE_RANK - 2, TRUE_RANK + 10])
@pytest.mark.parametrize('solver', ['full', 'arnoldi', 'wrong'])
@pytest.mark.parametrize('tikhonov_reg', [None, 0., 1e-5])
@pytest.mark.parametrize('observables', [None, lambda x: x[:, 0], np.random.rand(NUM_SAMPLES, 1)])
def test_KernelDMD_fit_predict_eig_modes_save_load(kernel, reduced_rank, rank, solver, tikhonov_reg, observables):
    
    dataset = make_linear_system()
    _Z = dataset.generate(np.zeros(DIM), NUM_SAMPLES)
    X, Y = _Z[:-1], _Z[1:]
    if solver not in ['full', 'arnoldi']:
        with pytest.raises(ValueError):
            model = KernelDMD(
                kernel=kernel, 
                reduced_rank=reduced_rank,
                rank=rank,
                tikhonov_reg=tikhonov_reg, 
                svd_solver=solver
                )
    else:
        model = KernelDMD(
                kernel=kernel, 
                reduced_rank=reduced_rank,
                rank=rank,
                tikhonov_reg=tikhonov_reg, 
                svd_solver=solver
                )

        assert model.is_fitted is False
        model.fit(X,Y)
        assert model.is_fitted is True
        if observables is None:
            X_pred = model.predict(X, observables=observables)
            assert X_pred.shape == X.shape
            modes = model.modes(X, observables=observables)
            assert modes.shape == (rank, ) + X.shape
        elif isinstance(observables, np.ndarray):
            X_pred = model.predict(X, observables=observables)
            assert X_pred.shape == observables.shape
            modes = model.modes(X, observables=observables)
            _target_shape = np.squeeze(np.zeros((rank, ) + observables.shape)).shape
            assert modes.shape == _target_shape
        else:
            _dummy_vec = observables(Y)
            if _dummy_vec.ndim == 1:
                _dummy_vec = _dummy_vec[:, None]
            X_pred = model.predict(X, observables=observables)
            assert X_pred.shape == _dummy_vec.shape
            modes = model.modes(X, observables=observables)
            _target_shape = np.squeeze(np.zeros((rank, ) + _dummy_vec.shape)).shape
            assert modes.shape == _target_shape

        vals, lv, rv = model.eig(eval_left_on=X, eval_right_on=X)
        assert vals.shape[0] == rank
        assert vals.ndim == 1
        tmp_path = Path(__file__).parent / f'tmp/model.bin'
        model.save(tmp_path)
        restored_model = KernelDMD.load(tmp_path)

        assert np.allclose(model.kernel_X, restored_model.kernel_X)
        assert np.allclose(model.kernel_Y, restored_model.kernel_Y)
        assert np.allclose(model.kernel_YX, restored_model.kernel_YX)
        assert np.allclose(model.X_fit, restored_model.X_fit)
        assert np.allclose(model.Y_fit, restored_model.Y_fit)
        rmtree(Path(__file__).parent / 'tmp/')