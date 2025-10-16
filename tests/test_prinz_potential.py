import pandas as pd

from kooplearn.datasets import make_prinz_potential


def test_triple_well_output_structure():
    df = make_prinz_potential(X0=0.0, n_steps=100, dt=1e-3, rng_seed=0)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x"]
    assert df.index.names == ["step", "time"]
    assert df.shape[0] == 101
    assert "generator" in df.attrs and df.attrs["generator"] == "make_prinz_potential"


def test_triple_well_reproducibility():
    df1 = make_prinz_potential(X0=0.0, n_steps=50, dt=1e-3, rng_seed=42)
    df2 = make_prinz_potential(X0=0.0, n_steps=50, dt=1e-3, rng_seed=42)
    pd.testing.assert_frame_equal(df1, df2)
