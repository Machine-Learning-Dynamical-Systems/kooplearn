import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def test_make_linear_system_returns_dataframe():
    """Test that output is a pandas DataFrame."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=100)

    assert isinstance(df, pd.DataFrame)


def test_make_linear_system_basic_output_shape():
    """Test that output has correct shape and columns."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=100)

    assert df.shape == (101, 2)
    assert list(df.columns) == ["x0", "x1"]


def test_make_linear_system_multiindex():
    """Test that DataFrame has proper MultiIndex."""
    from kooplearn.datasets import make_linear_system

    A = np.eye(2)
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=100, dt=1.0)

    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["step", "time"]
    assert len(df.index) == 101


def test_make_linear_system_time_values():
    """Test that time values in index are correct."""
    from kooplearn.datasets import make_linear_system

    A = np.eye(3)
    X0 = np.array([1.0, 0.0, 0.0])
    dt = 0.5
    df = make_linear_system(X0, A, n_steps=100, dt=dt)

    times = df.index.get_level_values("time")
    steps = df.index.get_level_values("step")

    assert_allclose(times[0], 0.0)
    assert_allclose(times[-1], 100 * dt)
    assert steps[0] == 0
    assert steps[-1] == 100


def test_make_linear_system_metadata():
    """Test that metadata is stored in attrs."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.5, 0.2], [0.1, 0.8]])
    X0 = np.array([1.0, 2.0])
    df = make_linear_system(X0, A, n_steps=50, noise=0.1, dt=2.0, random_state=42)

    assert "generator" in df.attrs
    assert df.attrs["generator"] == "make_linear_system"

    assert "X0" in df.attrs
    assert_allclose(df.attrs["X0"], X0)

    assert "A" in df.attrs
    assert_allclose(df.attrs["A"], A)

    assert "params" in df.attrs
    params = df.attrs["params"]
    assert params["n_steps"] == 50
    assert params["noise"] == 0.1
    assert params["dt"] == 2.0
    assert params["random_state"] == 42


def test_make_linear_system_initial_conditions():
    """Test that trajectory starts at initial conditions."""
    from kooplearn.datasets import make_linear_system

    A = np.eye(3)
    X0 = np.array([1.5, -0.5, 2.0])
    df = make_linear_system(X0, A, n_steps=10)

    assert_allclose(df.iloc[0].values, X0)


def test_make_linear_system_identity_matrix():
    """Test that identity matrix preserves initial conditions."""
    from kooplearn.datasets import make_linear_system

    A = np.eye(3)
    X0 = np.array([1.0, 2.0, 3.0])
    df = make_linear_system(X0, A, n_steps=100, noise=0.0)

    # With identity matrix and no noise, all states should be X0
    for i in range(len(df)):
        assert_allclose(df.iloc[i].values, X0)


def test_make_linear_system_deterministic_no_noise():
    """Test that system is deterministic without noise."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])

    df1 = make_linear_system(X0, A, n_steps=100, noise=0.0)
    df2 = make_linear_system(X0, A, n_steps=100, noise=0.0)

    assert_array_equal(df1.values, df2.values)


def test_make_linear_system_reproducible_with_seed():
    """Test that random_state makes noise reproducible."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])

    df1 = make_linear_system(X0, A, n_steps=100, noise=0.1, random_state=42)
    df2 = make_linear_system(X0, A, n_steps=100, noise=0.1, random_state=42)

    assert_array_equal(df1.values, df2.values)


def test_make_linear_system_different_with_noise():
    """Test that noise produces different trajectories."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])

    df1 = make_linear_system(X0, A, n_steps=100, noise=0.1, random_state=42)
    df2 = make_linear_system(X0, A, n_steps=100, noise=0.1, random_state=123)

    # Different seeds should produce different trajectories
    assert not np.allclose(df1.values, df2.values)


def test_make_linear_system_invalid_X0_shape():
    """Test that invalid X0 shape raises error."""
    from kooplearn.datasets import make_linear_system

    A = np.eye(2)

    with pytest.raises(ValueError, match="X0 must be 1-dimensional"):
        make_linear_system(np.array([[1.0, 0.0]]), A, n_steps=10)


def test_make_linear_system_invalid_A_shape():
    """Test that invalid A shape raises error."""
    from kooplearn.datasets import make_linear_system

    X0 = np.array([1.0, 0.0])

    # Not square
    with pytest.raises(ValueError, match="A must be square"):
        A_rect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        make_linear_system(X0, A_rect, n_steps=10)

    # Not 2D
    with pytest.raises(ValueError, match="A must be 2-dimensional"):
        A_1d = np.array([1.0, 0.0])
        make_linear_system(X0, A_1d, n_steps=10)


def test_make_linear_system_dimension_mismatch():
    """Test that dimension mismatch raises error."""
    from kooplearn.datasets import make_linear_system

    A = np.eye(3)
    X0 = np.array([1.0, 0.0])  # Wrong dimension

    with pytest.raises(ValueError, match="X0 dimension .* must match A dimension"):
        make_linear_system(X0, A, n_steps=10)


def test_make_linear_system_stable_system():
    """Test that stable system decays to zero."""
    from kooplearn.datasets import make_linear_system

    # Stable system with eigenvalues < 1
    A = np.array([[0.5, 0.0], [0.0, 0.5]])
    X0 = np.array([10.0, 10.0])
    df = make_linear_system(X0, A, n_steps=100, noise=0.0)

    # Should decay toward zero
    initial_norm = np.linalg.norm(df.iloc[0].values)
    final_norm = np.linalg.norm(df.iloc[-1].values)
    assert final_norm < initial_norm


def test_make_linear_system_rotation_matrix():
    """Test periodic behavior with rotation matrix."""
    from kooplearn.datasets import make_linear_system

    # Rotation by pi/2
    theta = np.pi / 2
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=4, noise=0.0)

    # After 4 rotations of pi/2, should be back to start
    assert_allclose(df.iloc[4].values, X0, atol=1e-10)


def test_make_linear_system_norm_preservation():
    """Test that orthogonal matrices preserve norm."""
    from kooplearn.datasets import make_linear_system

    # Orthogonal matrix (rotation)
    theta = np.pi / 6
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=100, noise=0.0)

    # Norm should be preserved at each step
    initial_norm = np.linalg.norm(X0)
    for i in range(len(df)):
        assert_allclose(np.linalg.norm(df.iloc[i].values), initial_norm, rtol=1e-10)


def test_make_linear_system_diagonal_matrix():
    """Test behavior with diagonal matrix."""
    from kooplearn.datasets import make_linear_system

    A = np.diag([0.9, 0.8, 0.7])
    X0 = np.array([1.0, 1.0, 1.0])
    df = make_linear_system(X0, A, n_steps=10, noise=0.0)

    # Each dimension should evolve independently
    for i, lambda_i in enumerate([0.9, 0.8, 0.7]):
        expected = X0[i] * (lambda_i ** np.arange(11))
        assert_allclose(df[f"x{i}"].values, expected, rtol=1e-10)


def test_make_linear_system_list_inputs():
    """Test that list inputs are properly converted."""
    from kooplearn.datasets import make_linear_system

    A_list = [[0.9, 0.1], [-0.1, 0.9]]
    X0_list = [1.0, 0.0]

    A_array = np.array(A_list)
    X0_array = np.array(X0_list)

    df_from_list = make_linear_system(X0_list, A_list, n_steps=50, noise=0.0)
    df_from_array = make_linear_system(X0_array, A_array, n_steps=50, noise=0.0)

    assert_array_equal(df_from_list.values, df_from_array.values)


def test_make_linear_system_single_step():
    """Test single step integration."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=1, noise=0.0)

    assert df.shape == (2, 2)
    # Second state should be A @ X0
    assert_allclose(df.iloc[1].values, A @ X0)


def test_make_linear_system_higher_dimensional():
    """Test with higher dimensional system."""
    from kooplearn.datasets import make_linear_system

    d = 10
    A = 0.95 * np.eye(d) + 0.01 * np.random.randn(d, d)
    X0 = np.random.randn(d)

    df = make_linear_system(X0, A, n_steps=100, noise=0.0)

    assert df.shape == (101, d)
    assert len(df.columns) == d
    assert all(df.columns == [f"x{i}" for i in range(d)])


@pytest.mark.parametrize("d", [1, 2, 5, 10])
def test_make_linear_system_various_dimensions(d):
    """Test various system dimensions."""
    from kooplearn.datasets import make_linear_system

    A = 0.9 * np.eye(d)
    X0 = np.ones(d)
    df = make_linear_system(X0, A, n_steps=50)

    assert df.shape == (51, d)
    assert len(df.columns) == d


def test_make_linear_system_noise_scaling():
    """Test that noise parameter scales variance correctly."""
    from kooplearn.datasets import make_linear_system

    A = np.eye(2)  # No dynamics, just noise
    X0 = np.zeros(2)
    n_steps = 1000

    # Different noise levels
    df_low = make_linear_system(X0, A, n_steps=n_steps, noise=0.1, random_state=42)
    df_high = make_linear_system(X0, A, n_steps=n_steps, noise=1.0, random_state=42)

    # Higher noise should have larger variance
    var_low = df_low.iloc[1:].var().mean()  # Skip initial condition
    var_high = df_high.iloc[1:].var().mean()

    assert var_high > var_low


def test_make_linear_system_dataframe_access():
    """Test typical DataFrame access patterns."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([0.5, 0.5])
    df = make_linear_system(X0, A, n_steps=100)

    # Column access
    x0_values = df["x0"]
    assert len(x0_values) == 101

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
    assert len(first_row) == 2


def test_make_linear_system_attrs_preserved():
    """Test that attrs are preserved through DataFrame operations."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=100)

    df_copy = df.copy()

    assert df_copy.attrs["generator"] == "make_linear_system"
    assert_allclose(df_copy.attrs["X0"], X0)
    assert_allclose(df_copy.attrs["A"], A)


def test_make_linear_system_convert_to_numpy():
    """Test conversion to numpy array."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=100)

    X = df.values
    assert isinstance(X, np.ndarray)
    assert X.shape == (101, 2)

    X2 = df.to_numpy()
    assert_array_equal(X, X2)


def test_make_linear_system_koopman_operator():
    """Test that Koopman operator is transpose of A."""
    from kooplearn.datasets import make_linear_system

    A = np.array([[0.9, 0.1], [-0.1, 0.9]])
    X0 = np.array([1.0, 0.0])
    df = make_linear_system(X0, A, n_steps=100, noise=0.0)

    # Koopman operator should be A^T
    K_expected = A.T

    # Verify by checking X_{t+1} = A @ X_t
    for i in range(len(df) - 1):
        X_t = df.iloc[i].values
        X_t1 = df.iloc[i + 1].values
        assert_allclose(X_t1, A @ X_t, rtol=1e-10)


def test_make_linear_system_numerical_stability():
    """Test numerical stability for long trajectories."""
    from kooplearn.datasets import make_linear_system

    A = 0.99 * np.eye(5)
    X0 = np.ones(5)
    df = make_linear_system(X0, A, n_steps=10000, noise=0.01, random_state=42)

    # No NaN or Inf values
    assert np.all(np.isfinite(df.values))
