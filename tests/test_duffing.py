import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def test_make_duffing_returns_dataframe():
    """Test that output is a pandas DataFrame."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.0, 0.0])
    df = make_duffing(X0, n_steps=100)

    assert isinstance(df, pd.DataFrame)


def test_make_duffing_basic_output_shape():
    """Test that output has correct shape and columns."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.0, 0.0])
    df = make_duffing(X0, n_steps=100)

    assert df.shape == (101, 2)
    assert list(df.columns) == ["position", "velocity"]


def test_make_duffing_multiindex():
    """Test that DataFrame has proper MultiIndex."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.0, 0.0])
    df = make_duffing(X0, n_steps=100, dt=0.01)

    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["step", "time"]
    assert len(df.index) == 101


def test_make_duffing_time_values():
    """Test that time values in index are correct."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.0, 0.0])
    df = make_duffing(X0, n_steps=100, dt=0.01)

    times = df.index.get_level_values("time")
    steps = df.index.get_level_values("step")

    assert_allclose(times[0], 0.0)
    assert_allclose(times[-1], 1.0, rtol=1e-5)
    assert steps[0] == 0
    assert steps[-1] == 100


def test_make_duffing_metadata():
    """Test that metadata is stored in attrs."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.5, -0.3])
    df = make_duffing(X0, n_steps=50, dt=0.02, alpha=1.0, beta=0.1)

    assert "generator" in df.attrs
    assert df.attrs["generator"] == "make_duffing"

    assert "X0" in df.attrs
    assert_allclose(df.attrs["X0"], X0)

    assert "params" in df.attrs
    params = df.attrs["params"]
    assert params["n_steps"] == 50
    assert params["dt"] == 0.02
    assert params["alpha"] == 1.0
    assert params["beta"] == 0.1


def test_make_duffing_initial_conditions():
    """Test that trajectory starts at initial conditions."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([1.5, -0.5])
    df = make_duffing(X0, n_steps=10)

    assert_allclose(df.iloc[0].values, X0, rtol=1e-5)


def test_make_duffing_energy_conservation_undamped():
    """Test approximate energy conservation for undamped, undriven system."""
    from kooplearn.datasets import make_duffing

    # Undamped, undriven Duffing (delta=0, gamma=0)
    X0 = np.array([1.0, 0.0])
    df = make_duffing(
        X0, n_steps=1000, dt=0.01, delta=0.0, gamma=0.0, alpha=1.0, beta=0.1
    )

    # Energy: E = 0.5*v^2 + 0.5*alpha*x^2 + 0.25*beta*x^4
    def energy(x, v, alpha=1.0, beta=0.1):
        return 0.5 * v**2 + 0.5 * alpha * x**2 + 0.25 * beta * x**4

    E0 = energy(df.iloc[0]["position"], df.iloc[0]["velocity"])
    E_final = energy(df.iloc[-1]["position"], df.iloc[-1]["velocity"])

    # Energy should be approximately conserved (within numerical error)
    assert_allclose(E_final, E0, rtol=1e-2)


def test_make_duffing_different_dt():
    """Test that different time steps produce consistent results."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.1, 0.1])

    # Same total time, different dt
    df1 = make_duffing(X0, n_steps=50, dt=0.01)
    df2 = make_duffing(X0, n_steps=100, dt=0.005)

    # Final states should be close
    assert_allclose(df1.iloc[-1].values, df2.iloc[-1].values, rtol=1e-2)


def test_make_duffing_deterministic():
    """Test that repeated calls with same parameters give same results."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.5, -0.3])

    df1 = make_duffing(X0, n_steps=100, dt=0.01, alpha=0.5)
    df2 = make_duffing(X0, n_steps=100, dt=0.01, alpha=0.5)

    assert_array_equal(df1.values, df2.values)


def test_make_duffing_invalid_initial_conditions():
    """Test that invalid initial conditions raise appropriate errors."""
    from kooplearn.datasets import make_duffing

    # Wrong shape
    with pytest.raises(ValueError, match="X0 must have shape"):
        make_duffing(np.array([0.0]))

    with pytest.raises(ValueError, match="X0 must have shape"):
        make_duffing(np.array([0.0, 0.0, 0.0]))


def test_make_duffing_parameter_effects():
    """Test that different parameters produce different trajectories."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.1, 0.1])

    # Default parameters
    df_default = make_duffing(X0, n_steps=100)

    # Different alpha
    df_alpha = make_duffing(X0, n_steps=100, alpha=1.0)

    # Trajectories should be different
    assert not np.allclose(df_default.iloc[-1].values, df_alpha.iloc[-1].values)


def test_make_duffing_chaotic_regime():
    """Test trajectory in chaotic regime has expected properties."""
    from kooplearn.datasets import make_duffing

    # Known chaotic parameters
    X0 = np.array([0.1, 0.1])
    df = make_duffing(
        X0,
        n_steps=5000,
        dt=0.01,
        alpha=-1.0,
        beta=1.0,
        gamma=0.3,
        delta=0.25,
        omega=1.0,
    )

    # In chaotic regime, trajectory should explore state space
    assert df["position"].max() - df["position"].min() > 1.0
    assert df["velocity"].max() - df["velocity"].min() > 1.0


def test_make_duffing_list_input():
    """Test that list inputs are properly converted."""
    from kooplearn.datasets import make_duffing

    X0_list = [0.5, -0.3]
    X0_array = np.array(X0_list)

    df_from_list = make_duffing(X0_list, n_steps=50)
    df_from_array = make_duffing(X0_array, n_steps=50)

    assert_array_equal(df_from_list.values, df_from_array.values)


def test_make_duffing_single_step():
    """Test that single step integration works."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([1.0, 0.0])
    df = make_duffing(X0, n_steps=1, dt=0.1)

    assert df.shape == (2, 2)


def test_make_duffing_time_consistency():
    """Test that time array matches n_steps and dt."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.0, 0.0])
    dt = 0.05
    n_steps = 200

    df = make_duffing(X0, n_steps=n_steps, dt=dt)
    times = df.index.get_level_values("time").values

    assert len(times) == n_steps + 1
    assert_allclose(np.diff(times), dt, rtol=1e-6)


@pytest.mark.parametrize(
    "alpha,beta,gamma,delta,omega",
    [
        (0.5, 0.0625, 0.1, 2.5, 2.0),  # Default
        (1.0, 0.1, 0.2, 1.0, 1.5),  # Custom 1
        (-1.0, 1.0, 0.3, 0.25, 1.0),  # Chaotic
    ],
)
def test_make_duffing_parameter_combinations(alpha, beta, gamma, delta, omega):
    """Test various parameter combinations produce valid output."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.1, 0.1])
    df = make_duffing(
        X0, n_steps=100, alpha=alpha, beta=beta, gamma=gamma, delta=delta, omega=omega
    )

    assert df.shape == (101, 2)
    assert np.all(np.isfinite(df.values))


def test_make_duffing_zero_driving():
    """Test behavior with no external driving force."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([1.0, 0.0])
    df = make_duffing(X0, n_steps=500, gamma=0.0, delta=0.1)

    # With damping but no driving, system should approach equilibrium
    assert np.abs(df.iloc[-1]["position"]) < np.abs(df.iloc[0]["position"])
    assert np.abs(df.iloc[-1]["velocity"]) < 0.5


def test_make_duffing_numerical_stability():
    """Test that integration remains stable for long trajectories."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.1, 0.1])
    df = make_duffing(X0, n_steps=10000, dt=0.01)

    # No NaN or Inf values
    assert np.all(np.isfinite(df.values))

    # Values should remain bounded (for these parameters)
    assert np.abs(df.values).max() < 100


def test_make_duffing_dataframe_access():
    """Test typical DataFrame access patterns."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.5, 0.5])
    df = make_duffing(X0, n_steps=100)

    # Column access
    positions = df["position"]
    assert len(positions) == 101

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
    assert len(first_row) == 2


def test_make_duffing_attrs_preserved():
    """Test that attrs are preserved through DataFrame operations."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.1, 0.1])
    df = make_duffing(X0, n_steps=100)

    # Create a copy
    df_copy = df.copy()

    # Attrs should be preserved in copy
    assert df_copy.attrs["generator"] == "make_duffing"
    assert_allclose(df_copy.attrs["X0"], X0)


def test_make_duffing_convert_to_numpy():
    """Test conversion to numpy array."""
    from kooplearn.datasets import make_duffing

    X0 = np.array([0.1, 0.1])
    df = make_duffing(X0, n_steps=100)

    # Convert to numpy
    X = df.values
    assert isinstance(X, np.ndarray)
    assert X.shape == (101, 2)

    # Can also use .to_numpy()
    X2 = df.to_numpy()
    assert_array_equal(X, X2)
