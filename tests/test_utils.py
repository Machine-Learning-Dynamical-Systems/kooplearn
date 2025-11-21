import numpy as np
import pytest

from kooplearn._src.utils import parse_cplx_eig, topk

rng = np.random.default_rng(42)  # Global rng


@pytest.mark.parametrize("k", [0, 1, 3])
def test_topk(k):
    x = np.arange(10)
    if k == 0:
        with pytest.raises(AssertionError):
            res = topk(x, k=k)
    else:
        res = topk(x, k=k)
        assert np.allclose(res.values, np.flip(x)[:k])
        assert np.allclose(x[res.indices], np.flip(x)[:k])

    with pytest.raises(AssertionError):
        res = topk(rng.random((10, 20)), k=10)


@pytest.mark.parametrize("num_reals", [0, 1, 3])
@pytest.mark.parametrize("num_cplx_pairs", [0, 1, 3])
def test_parse_cplx_eig_(num_reals, num_cplx_pairs):
    rand_real = np.sort(rng.random(num_reals))

    rand_angle = rng.random(num_cplx_pairs) * np.pi
    rand_abs = rng.random(num_cplx_pairs)
    rand_cplx = np.sort(rand_abs * np.exp(1j * rand_angle))

    vec = np.concatenate([rand_real, rand_cplx, rand_cplx.conj()])

    rand_perm = rng.permutation(vec.shape[0])
    assert np.allclose(
        parse_cplx_eig(vec[rand_perm]), vec[: (num_reals + num_cplx_pairs)]
    )
