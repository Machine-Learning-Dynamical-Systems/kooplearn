import pytest
from scipy.stats import special_ortho_group
from shutil import rmtree
import numpy as np
from pathlib import Path
from kooplearn._src.context_window_utils import trajectory_to_contexts
from kooplearn.models import KernelDMD
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern
from kooplearn.datasets.stochastic import LinearModel


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
@pytest.mark.parametrize('observables', [None, lambda x: x[:, 0], 'array'])
@pytest.mark.parametrize('lookback_len', [1, 2, 3])
def test_KernelDMD_fit_predict_eig_modes_save_load(kernel, reduced_rank, rank, solver, tikhonov_reg, observables, lookback_len):
    
    dataset = make_linear_system()
    _Z = dataset.generate(np.zeros(DIM), NUM_SAMPLES)
    data = trajectory_to_contexts(_Z, lookback_len + 1)
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
        model.fit(data, lookback_len=lookback_len)
        assert model.is_fitted is True
        if observables is None:
            X_pred = model.predict(data, observables=observables)
            assert X_pred.shape == (data.shape[0],) + data.shape[2:]
            modes = model.modes(data, observables=observables)
            assert modes.shape == (rank, ) + (data.shape[0],) + data.shape[2:]
        elif isinstance(observables, np.ndarray):
            assert observables == 'array'
            observables = np.random.rand(data.shape[0], 1, 2, 3)
            X_pred = model.predict(data, observables=observables)
            assert X_pred.shape == observables.shape
            modes = model.modes(data, observables=observables)
            _target_shape = np.squeeze(np.zeros((rank, ) + observables.shape)).shape
            assert modes.shape == _target_shape
        else:
            Y = data[:, -1, ...]
            _dummy_vec = observables(Y)
            if _dummy_vec.ndim == 1:
                _dummy_vec = _dummy_vec[:, None]
            X_pred = model.predict(data, observables=observables)
            assert X_pred.shape == _dummy_vec.shape
            modes = model.modes(data, observables=observables)
            _target_shape = np.squeeze(np.zeros((rank, ) + _dummy_vec.shape)).shape
            assert modes.shape == _target_shape

        vals, lv, rv = model.eig(eval_left_on=data, eval_right_on=data)
        assert vals.shape[0] == rank
        assert vals.ndim == 1
        tmp_path = Path(__file__).parent / f'tmp/model.bin'
        model.save(tmp_path)
        restored_model = KernelDMD.load(tmp_path)

        assert np.allclose(model.kernel_X, restored_model.kernel_X)
        assert np.allclose(model.kernel_Y, restored_model.kernel_Y)
        assert np.allclose(model.kernel_YX, restored_model.kernel_YX)
        assert np.allclose(model.data_fit, restored_model.data_fit)
        assert np.allclose(model._lookback_len, model._lookback_len)
        rmtree(Path(__file__).parent / 'tmp/')