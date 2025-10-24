import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def test_make_logistic_map_returns_dataframe():
    """Test that output is a pandas DataFrame."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.1, n_steps=100)

    assert isinstance(df, pd.DataFrame)


def test_make_logistic_map_basic_output_shape():
    """Test that output has correct shape and columns."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.1, n_steps=100)

    assert df.shape == (101, 1)
    assert list(df.columns) == ["x"]


def test_make_logistic_map_multiindex():
    """Test that DataFrame has proper MultiIndex."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.1, n_steps=100, dt=1.0)

    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["step", "time"]
    assert len(df.index) == 101


def test_make_logistic_map_time_values():
    """Test that time values in index are correct."""
    from kooplearn.datasets import make_logistic_map

    dt = 0.5
    df = make_logistic_map(X0=0.1, n_steps=100, dt=dt)

    times = df.index.get_level_values("time")
    steps = df.index.get_level_values("step")

    assert_allclose(times[0], 0.0)
    assert_allclose(times[-1], 100 * dt)
    assert steps[0] == 0
    assert steps[-1] == 100


def test_make_logistic_map_metadata():
    """Test that metadata is stored in attrs."""
    from kooplearn.datasets import make_logistic_map

    X0 = 0.3
    df = make_logistic_map(X0=X0, n_steps=50, r=3.8, M=15, random_state=42)

    assert "generator" in df.attrs
    assert df.attrs["generator"] == "make_logistic_map"

    assert "X0" in df.attrs
    assert_allclose(df.attrs["X0"], X0)

    assert "params" in df.attrs
    params = df.attrs["params"]
    assert params["n_steps"] == 50
    assert params["r"] == 3.8
    assert params["M"] == 15
    assert params["random_state"] == 42


def test_make_logistic_map_initial_condition():
    """Test that trajectory starts at initial condition."""
    from kooplearn.datasets import make_logistic_map

    X0 = 0.7
    df = make_logistic_map(X0=X0, n_steps=10)

    assert_allclose(df.iloc[0].values[0], X0)


def test_make_logistic_map_chaotic_dynamics():
    """Test chaotic dynamics for r=4."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.1, n_steps=1000, r=4.0)

    # Should explore full [0, 1] range
    assert df["x"].min() < 0.1
    assert df["x"].max() > 0.9


def test_make_logistic_map_deterministic_no_noise():
    """Test that system is deterministic without noise."""
    from kooplearn.datasets import make_logistic_map

    X0 = 0.1
    df1 = make_logistic_map(X0=X0, n_steps=100, r=4.0, M=0)
    df2 = make_logistic_map(X0=X0, n_steps=100, r=4.0, M=0)

    assert_array_equal(df1.values, df2.values)


def test_make_logistic_map_reproducible_with_seed():
    """Test that random_state makes noise reproducible."""
    from kooplearn.datasets import make_logistic_map

    X0 = 0.1
    df1 = make_logistic_map(X0=X0, n_steps=100, r=4.0, random_state=42)
    df2 = make_logistic_map(X0=X0, n_steps=100, r=4.0, random_state=42)

    assert_array_equal(df1.values, df2.values)


def test_make_logistic_map_different_with_noise():
    """Test that noise produces different trajectories."""
    from kooplearn.datasets import make_logistic_map

    X0 = 0.1
    df1 = make_logistic_map(X0=X0, n_steps=100, r=4.0, random_state=42)
    df2 = make_logistic_map(X0=X0, n_steps=100, r=4.0, random_state=123)

    # Different seeds should produce different trajectories
    assert not np.allclose(df1.values, df2.values)


def test_make_logistic_map_invalid_X0():
    """Test that invalid X0 raises error."""
    from kooplearn.datasets import make_logistic_map

    # Out of bounds
    with pytest.raises(ValueError, match="X0 must be in \\[0, 1\\]"):
        make_logistic_map(X0=-0.1, n_steps=10)

    with pytest.raises(ValueError, match="X0 must be in \\[0, 1\\]"):
        make_logistic_map(X0=1.5, n_steps=10)

    # Wrong shape
    with pytest.raises(ValueError, match="X0 must be scalar"):
        make_logistic_map(X0=[0.1, 0.2], n_steps=10)


def test_make_logistic_map_bounded_values():
    """Test that values stay in [0, 1] with clipping."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.5, n_steps=100, r=4.0, random_state=42)

    # All values should be in [0, 1]
    assert np.all(df["x"].values >= 0)
    assert np.all(df["x"].values <= 1)


def test_make_logistic_map_fixed_points():
    """Test behavior at fixed points."""
    from kooplearn.datasets import make_logistic_map

    # For r < 1, x=0 is stable fixed point
    df = make_logistic_map(X0=0.01, n_steps=50, r=0.5, M=0)
    assert df.iloc[-1]["x"] < 0.001  # Should converge to 0

    # For 1 < r < 3, x=(r-1)/r is stable
    r = 2.5
    fixed_point = (r - 1) / r
    df = make_logistic_map(X0=fixed_point, n_steps=10, r=r, M=0)
    # Should stay at fixed point
    assert_allclose(df["x"].values, fixed_point, atol=1e-10)


def test_make_logistic_map_period_doubling():
    """Test period-2 orbit in appropriate regime."""
    from kooplearn.datasets import make_logistic_map

    # Around r ≈ 3.2, should have period-2 orbit
    df = make_logistic_map(X0=0.5, n_steps=100, r=3.2, M=0)

    # After transient, should oscillate between two values
    steady = df["x"].iloc[-20:].values
    unique_vals = np.unique(np.round(steady, decimals=5))
    assert len(unique_vals) == 2  # Period-2


def test_make_logistic_map_sensitive_dependence():
    """Test sensitive dependence on initial conditions."""
    from kooplearn.datasets import make_logistic_map

    # Two very close initial conditions
    df1 = make_logistic_map(X0=0.1, n_steps=100, r=4.0)
    df2 = make_logistic_map(X0=0.1 + 1e-8, n_steps=100, r=4.0)

    # Initially close
    assert_allclose(df1.iloc[0]["x"], df2.iloc[0]["x"], atol=1e-7)

    # Should diverge significantly (Lyapunov exponent > 0)
    distance = (
        2
        * np.abs(df1.iloc[-1]["x"] - df2.iloc[-1]["x"])
        / (df1.iloc[-1]["x"] + df2.iloc[-1]["x"])
    )
    assert distance > 0.1


def test_make_logistic_map_array_X0():
    """Test that array X0 is properly handled."""
    from kooplearn.datasets import make_logistic_map

    X0_float = 0.5
    X0_array = np.array([0.5])

    df_from_float = make_logistic_map(X0=X0_float, n_steps=50, M=0)
    df_from_array = make_logistic_map(X0=X0_array, n_steps=50, M=0)

    assert_array_equal(df_from_float.values, df_from_array.values)


def test_make_logistic_map_single_step():
    """Test single step integration."""
    from kooplearn.datasets import make_logistic_map

    X0 = 0.3
    r = 4.0
    df = make_logistic_map(X0=X0, n_steps=1, r=r, M=0)

    assert df.shape == (2, 1)
    # Second state should be r * X0 * (1 - X0)
    expected = r * X0 * (1 - X0)
    assert_allclose(df.iloc[1]["x"], expected)


@pytest.mark.parametrize("r", [0.5, 2.0, 3.2, 3.57, 4.0])
def test_make_logistic_map_various_r_values(r):
    """Test various growth rate parameters."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.1, n_steps=100, r=r)

    assert df.shape == (101, 1)
    assert np.all(np.isfinite(df.values))
    assert np.all(df["x"].values >= 0)
    assert np.all(df["x"].values <= 1)


@pytest.mark.parametrize("M", [5, 10, 20])
def test_make_logistic_map_various_noise_orders(M):
    """Test various noise orders."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.1, n_steps=100, r=4.0, M=M, random_state=42)

    assert df.shape == (101, 1)
    assert np.all(np.isfinite(df.values))


def test_make_logistic_map_noise_distribution():
    """Test that noise has correct distribution properties."""
    from kooplearn.datasets import make_logistic_map

    # Generate many trajectories and look at noise
    n_samples = 10000
    df = make_logistic_map(X0=0.5, n_steps=n_samples, r=1.0, M=10, random_state=42)

    # For r=1, x_t+1 = x_t + noise, so differences reveal noise
    # (approximately, up to the x_t*(1-x_t) term which is ≈ 1/4)
    differences = df["x"].diff().iloc[1:].values

    # Noise should be approximately centered at 0.25 (from 1*x*(1-x) with x≈0.5)
    # and bounded in [-0.5*noise, 0.5*noise] range
    assert np.abs(differences).max() < 0.3  # Rough bound


def test_make_logistic_map_dataframe_access():
    """Test typical DataFrame access patterns."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.5, n_steps=100)

    # Column access
    x_values = df["x"]
    assert len(x_values) == 101

    # Index slicing by step
    first_10_steps = df.loc[0:9]
    assert len(first_10_steps) == 10

    # Access specific step
    step_50 = df.xs(50, level="step")
    assert isinstance(step_50, pd.DataFrame)
    assert len(step_50) == 1

    # Access by position with iloc
    first_row = df.iloc[0]
    assert isinstance(first_row, pd.Series)
    assert len(first_row) == 1


def test_make_logistic_map_attrs_preserved():
    """Test that attrs are preserved through DataFrame operations."""
    from kooplearn.datasets import make_logistic_map

    X0 = 0.1
    df = make_logistic_map(X0=X0, n_steps=100)

    df_copy = df.copy()

    assert df_copy.attrs["generator"] == "make_logistic_map"
    assert_allclose(df_copy.attrs["X0"], X0)


def test_make_logistic_map_convert_to_numpy():
    """Test conversion to numpy array."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.1, n_steps=100)

    X = df.values
    assert isinstance(X, np.ndarray)
    assert X.shape == (101, 1)

    X2 = df.to_numpy()
    assert_array_equal(X, X2)


def test_make_logistic_map_numerical_stability():
    """Test numerical stability for long trajectories."""
    from kooplearn.datasets import make_logistic_map

    df = make_logistic_map(X0=0.1, n_steps=10000, r=4.0, random_state=42)

    # No NaN or Inf values
    assert np.all(np.isfinite(df.values))

    # All values in valid range
    assert np.all(df["x"].values >= 0)
    assert np.all(df["x"].values <= 1)


def test_make_logistic_map_ergodic_measure():
    """Test that chaotic map approximates invariant measure."""
    from kooplearn.datasets import make_logistic_map

    # For r=4, invariant density is 1/(pi*sqrt(x*(1-x)))
    df = make_logistic_map(X0=0.1, n_steps=10000, r=4.0)

    # After transient, histogram should roughly match invariant measure
    x_steady = df["x"].iloc[1000:].values

    # Invariant measure concentrates near 0 and 1
    bins_near_edges = ((x_steady < 0.1) | (x_steady > 0.9)).sum()
    bins_middle = ((x_steady >= 0.4) & (x_steady <= 0.6)).sum()

    # Should have more samples near edges
    assert bins_near_edges > bins_middle
