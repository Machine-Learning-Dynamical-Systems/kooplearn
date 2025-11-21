import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def test_make_lorenz63_returns_dataframe():
    """Test that output is a pandas DataFrame."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([1.0, 0.0, 0.0])
    df = make_lorenz63(X0, n_steps=100)

    assert isinstance(df, pd.DataFrame)


def test_make_lorenz63_basic_output_shape():
    """Test that output has correct shape and columns."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([1.0, 0.0, 0.0])
    df = make_lorenz63(X0, n_steps=100)

    assert df.shape == (101, 3)
    assert list(df.columns) == ["x", "y", "z"]


def test_make_lorenz63_multiindex():
    """Test that DataFrame has proper MultiIndex."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([1.0, 0.0, 0.0])
    df = make_lorenz63(X0, n_steps=100, dt=0.01)

    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["step", "time"]
    assert len(df.index) == 101


def test_make_lorenz63_time_values():
    """Test that time values in index are correct."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([1.0, 0.0, 0.0])
    df = make_lorenz63(X0, n_steps=100, dt=0.01)

    times = df.index.get_level_values("time")
    steps = df.index.get_level_values("step")

    assert_allclose(times[0], 0.0)
    assert_allclose(times[-1], 1.0, rtol=1e-5)
    assert steps[0] == 0
    assert steps[-1] == 100


def test_make_lorenz63_metadata():
    """Test that metadata is stored in attrs."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.5, -0.3, 1.2])
    df = make_lorenz63(X0, n_steps=50, dt=0.02, sigma=12.0, mu=30.0, beta=3.0)

    assert "generator" in df.attrs
    assert df.attrs["generator"] == "make_lorenz63"

    assert "X0" in df.attrs
    assert_allclose(df.attrs["X0"], X0)

    assert "params" in df.attrs
    params = df.attrs["params"]
    assert params["n_steps"] == 50
    assert params["dt"] == 0.02
    assert params["sigma"] == 12.0
    assert params["mu"] == 30.0
    assert params["beta"] == 3.0


def test_make_lorenz63_initial_conditions():
    """Test that trajectory starts at initial conditions."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([1.5, -0.5, 2.0])
    df = make_lorenz63(X0, n_steps=10)

    assert_allclose(df.iloc[0].values, X0, rtol=1e-5)


def test_make_lorenz63_deterministic():
    """Test that repeated calls with same parameters give same results."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.5, -0.3, 1.0])

    df1 = make_lorenz63(X0, n_steps=100, dt=0.01, sigma=10.0)
    df2 = make_lorenz63(X0, n_steps=100, dt=0.01, sigma=10.0)

    assert_array_equal(df1.values, df2.values)


def test_make_lorenz63_invalid_initial_conditions():
    """Test that invalid initial conditions raise appropriate errors."""
    from kooplearn.datasets import make_lorenz63

    # Wrong shape
    with pytest.raises(ValueError, match="X0 must have shape"):
        make_lorenz63(np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="X0 must have shape"):
        make_lorenz63(np.array([0.0, 0.0, 0.0, 0.0]))


def test_make_lorenz63_parameter_effects():
    """Test that different parameters produce different trajectories."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.1, 0.1, 0.1])

    # Default parameters
    df_default = make_lorenz63(X0, n_steps=100)

    # Different mu (Rayleigh number)
    df_mu = make_lorenz63(X0, n_steps=100, mu=20.0)

    # Trajectories should be different
    assert not np.allclose(df_default.iloc[-1].values, df_mu.iloc[-1].values)


def test_make_lorenz63_chaotic_behavior():
    """Test that system exhibits chaotic behavior with classic parameters."""
    from kooplearn.datasets import make_lorenz63

    # Classic chaotic parameters
    X0 = np.array([0.0, 1.0, 1.05])
    df = make_lorenz63(X0, n_steps=5000, dt=0.01, sigma=10.0, mu=28.0, beta=8.0 / 3.0)

    # In chaotic regime, trajectory should explore wide state space
    assert df["x"].max() - df["x"].min() > 20.0
    assert df["y"].max() - df["y"].min() > 30.0
    assert df["z"].max() - df["z"].min() > 30.0


def test_make_lorenz63_butterfly_attractor():
    """Test that trajectory stays on the butterfly attractor."""
    from kooplearn.datasets import make_lorenz63

    # Classic parameters for butterfly attractor
    X0 = np.array([1.0, 1.0, 1.0])
    df = make_lorenz63(X0, n_steps=10000, dt=0.01)

    # After initial transient, should be on attractor
    # z values should be mostly positive and bounded
    z_stable = df["z"].iloc[1000:]
    assert z_stable.min() > -5.0
    assert z_stable.max() < 60.0


def test_make_lorenz63_sensitive_dependence():
    """Test sensitive dependence on initial conditions."""
    from kooplearn.datasets import make_lorenz63

    # Two very close initial conditions
    X0_1 = np.array([1.0, 0.0, 0.0])
    X0_2 = np.array([1.0 + 1e-8, 0.0, 0.0])

    df1 = make_lorenz63(X0_1, n_steps=1000, dt=0.05)
    df2 = make_lorenz63(X0_2, n_steps=1000, dt=0.05)

    # Initially close
    assert_allclose(df1.iloc[0].values, df2.iloc[0].values, atol=1e-7)

    # Should diverge significantly after some time (chaos)
    distance = np.linalg.norm(df1.iloc[-1].values - df2.iloc[-1].values)
    assert distance > 1.0


def test_make_lorenz63_list_input():
    """Test that list inputs are properly converted."""
    from kooplearn.datasets import make_lorenz63

    X0_list = [0.5, -0.3, 1.2]
    X0_array = np.array(X0_list)

    df_from_list = make_lorenz63(X0_list, n_steps=50)
    df_from_array = make_lorenz63(X0_array, n_steps=50)

    assert_array_equal(df_from_list.values, df_from_array.values)


def test_make_lorenz63_single_step():
    """Test that single step integration works."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([1.0, 0.0, 0.0])
    df = make_lorenz63(X0, n_steps=1, dt=0.1)

    assert df.shape == (2, 3)


def test_make_lorenz63_time_consistency():
    """Test that time array matches n_steps and dt."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([1.0, 0.0, 0.0])
    dt = 0.05
    n_steps = 200

    df = make_lorenz63(X0, n_steps=n_steps, dt=dt)
    times = df.index.get_level_values("time").values

    assert len(times) == n_steps + 1
    assert_allclose(np.diff(times), dt, rtol=1e-6)


def test_make_lorenz63_different_dt():
    """Test that different time steps produce consistent results."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.1, 0.1, 0.1])

    # Same total time, different dt
    df1 = make_lorenz63(X0, n_steps=100, dt=0.01)
    df2 = make_lorenz63(X0, n_steps=200, dt=0.005)

    # Final states should be close
    assert_allclose(df1.iloc[-1].values, df2.iloc[-1].values, rtol=1e-2)


@pytest.mark.parametrize(
    "sigma,mu,beta",
    [
        (10.0, 28.0, 8.0 / 3.0),  # Classic chaotic
        (10.0, 20.0, 8.0 / 3.0),  # Different mu
        (12.0, 28.0, 8.0 / 3.0),  # Different sigma
        (10.0, 28.0, 3.0),  # Different beta
    ],
)
def test_make_lorenz63_parameter_combinations(sigma, mu, beta):
    """Test various parameter combinations produce valid output."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.1, 0.1, 0.1])
    df = make_lorenz63(X0, n_steps=100, sigma=sigma, mu=mu, beta=beta)

    assert df.shape == (101, 3)
    assert np.all(np.isfinite(df.values))


def test_make_lorenz63_fixed_points():
    """Test behavior near fixed points."""
    from kooplearn.datasets import make_lorenz63

    # For mu > 1, there are non-trivial fixed points at
    # (±sqrt(beta*(mu-1)), ±sqrt(beta*(mu-1)), mu-1)
    mu = 28.0
    beta = 8.0 / 3.0
    sigma = 10.0

    # Start very close to origin (unstable fixed point for mu > 1)
    X0 = np.array([1e-8, 0.0, 0.0])
    df = make_lorenz63(X0, n_steps=1000, sigma=sigma, mu=mu, beta=beta, dt=0.05)

    # Should move away from origin
    final_distance = np.linalg.norm(df.iloc[-1].values)
    assert final_distance > 1.0


def test_make_lorenz63_numerical_stability():
    """Test that integration remains stable for long trajectories."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.1, 0.1, 0.1])
    df = make_lorenz63(X0, n_steps=20000, dt=0.01)

    # No NaN or Inf values
    assert np.all(np.isfinite(df.values))

    # Values should remain bounded on the attractor
    assert np.abs(df.values).max() < 100


def test_make_lorenz63_dataframe_access():
    """Test typical DataFrame access patterns."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.5, 0.5, 0.5])
    df = make_lorenz63(X0, n_steps=100)

    # Column access
    x_values = df["x"]
    assert len(x_values) == 101

    # Index slicing by step
    first_10_steps = df.loc[0:9]
    assert len(first_10_steps) == 10

    # Access specific step (returns DataFrame with time index)
    step_50 = df.xs(50, level="step")
    assert isinstance(step_50, pd.DataFrame)
    assert len(step_50) == 1

    # Access by position with iloc
    first_row = df.iloc[0]
    assert isinstance(first_row, pd.Series)
    assert len(first_row) == 3


def test_make_lorenz63_attrs_preserved():
    """Test that attrs are preserved through DataFrame operations."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.1, 0.1, 0.1])
    df = make_lorenz63(X0, n_steps=100)

    # Create a copy
    df_copy = df.copy()

    # Attrs should be preserved in copy
    assert df_copy.attrs["generator"] == "make_lorenz63"
    assert_allclose(df_copy.attrs["X0"], X0)


def test_make_lorenz63_convert_to_numpy():
    """Test conversion to numpy array."""
    from kooplearn.datasets import make_lorenz63

    X0 = np.array([0.1, 0.1, 0.1])
    df = make_lorenz63(X0, n_steps=100)

    # Convert to numpy
    X = df.values
    assert isinstance(X, np.ndarray)
    assert X.shape == (101, 3)

    # Can also use .to_numpy()
    X2 = df.to_numpy()
    assert_array_equal(X, X2)


def test_make_lorenz63_subcritical_regime():
    """Test behavior in subcritical regime (mu < 1)."""
    from kooplearn.datasets import make_lorenz63

    # Subcritical: should converge to origin
    X0 = np.array([5.0, 5.0, 5.0])
    df = make_lorenz63(X0, n_steps=1000, dt=0.01, mu=0.5)

    # Should approach origin
    final_distance = np.linalg.norm(df.iloc[-1].values)
    initial_distance = np.linalg.norm(df.iloc[0].values)
    assert final_distance < initial_distance
