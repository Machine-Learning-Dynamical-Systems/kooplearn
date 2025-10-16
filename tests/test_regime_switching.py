import numpy as np
import pandas as pd
import pytest

from kooplearn.datasets import make_regime_switching_var


def test_output_structure_and_metadata():
    phi1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    phi2 = np.array([[0.5, -0.2], [0.3, 0.7]])
    transition = np.array([[0.95, 0.05], [0.1, 0.9]])
    X0 = np.zeros(2)

    df = make_regime_switching_var(
        X0, phi1, phi2, transition, n_steps=50, noise=0.01, rng_seed=42
    )

    # Check types and shape
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (51, 2)
    assert list(df.columns) == ["x0", "x1"]

    # Check MultiIndex
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["step", "time"]

    # Metadata
    attrs = df.attrs
    assert attrs["generator"] == "make_regime_switching_var"
    assert "params" in attrs
    assert "regimes" in attrs
    assert len(attrs["regimes"]) == 51


def test_reproducibility_with_seed():
    phi1 = np.eye(2)
    phi2 = np.eye(2)
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    X0 = np.zeros(2)

    df1 = make_regime_switching_var(
        X0, phi1, phi2, transition, n_steps=20, noise=0.1, rng_seed=123
    )
    df2 = make_regime_switching_var(
        X0, phi1, phi2, transition, n_steps=20, noise=0.1, rng_seed=123
    )

    pd.testing.assert_frame_equal(df1, df2)
    np.testing.assert_array_equal(df1.attrs["regimes"], df2.attrs["regimes"])


def test_invalid_transition_matrix_shape_and_rowsum():
    phi1 = np.eye(2)
    phi2 = np.eye(2)
    X0 = np.zeros(2)

    with pytest.raises(ValueError, match="must be a 2x2 matrix"):
        make_regime_switching_var(X0, phi1, phi2, np.ones((3, 3)), n_steps=10)

    bad_transition = np.array([[0.6, 0.6], [0.5, 0.6]])
    with pytest.raises(ValueError, match="must sum to 1"):
        make_regime_switching_var(X0, phi1, phi2, bad_transition, n_steps=10)


def test_custom_initial_condition():
    phi1 = np.eye(2)
    phi2 = np.eye(2)
    transition = np.array([[0.95, 0.05], [0.05, 0.95]])
    X0 = np.array([1.0, -1.0])

    df = make_regime_switching_var(
        X0, phi1, phi2, transition, n_steps=5, noise=0.0, rng_seed=0
    )
    np.testing.assert_array_almost_equal(df.iloc[0].values, X0)


def test_transition_probabilities_empirical():
    phi1 = np.eye(1)
    phi2 = np.eye(1)
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    X0 = np.zeros(1)

    df = make_regime_switching_var(
        X0, phi1, phi2, transition, n_steps=10000, noise=0.0, rng_seed=0
    )
    regimes = df.attrs["regimes"]

    counts = np.zeros((2, 2))
    for i in range(len(regimes) - 1):
        counts[regimes[i], regimes[i + 1]] += 1
    empirical = counts / counts.sum(axis=1, keepdims=True)

    assert np.allclose(empirical, transition, atol=0.05)
