import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from kooplearn.jax.utils import NnxFeatureMapEmbedder


# Simple NNX test modules
class SimpleEncoder(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


class SimpleDecoder(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


@pytest.fixture
def encoder(rngs):
    return SimpleEncoder(in_features=10, out_features=5, rngs=rngs)


@pytest.fixture
def decoder(rngs):
    return SimpleDecoder(in_features=5, out_features=10, rngs=rngs)


@pytest.fixture
def sample_data():
    return np.random.randn(20, 10).astype(np.float32)


def test_init_encoder_only(encoder):
    """Test initialization with encoder only."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    assert embedder.encoder is encoder
    assert embedder.decoder is None
    assert embedder.jitted_encoder_call is not None
    assert embedder.jitted_decoder_call is None


def test_init_encoder_decoder(encoder, decoder):
    """Test initialization with both encoder and decoder."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder, decoder=decoder)
    assert embedder.encoder is encoder
    assert embedder.decoder is decoder
    assert embedder.jitted_encoder_call is not None
    assert embedder.jitted_decoder_call is not None


def test_fit_returns_self(encoder, sample_data):
    """Test that fit() returns self (sklearn API compliance)."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    result = embedder.fit(sample_data)
    assert result is embedder


def test_fit_with_y(encoder, sample_data):
    """Test that fit() accepts y parameter (sklearn API)."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    y = np.random.randn(20, 5)
    result = embedder.fit(sample_data, y=y)
    assert result is embedder


def test_fit_with_none(encoder):
    """Test that fit() works with None inputs."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    result = embedder.fit()
    assert result is embedder


def test_transform_numpy_input(encoder, sample_data):
    """Test transform with numpy array input."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit(sample_data)
    encoded = embedder.transform(sample_data)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (20, 5)
    assert encoded.dtype == np.float32


def test_transform_jax_array_input(encoder, sample_data):
    """Test transform with JAX array input."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit(sample_data)

    jax_input = jnp.asarray(sample_data)
    encoded = embedder.transform(jax_input)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (20, 5)


def test_transform_output_consistency(encoder, sample_data):
    """Test that multiple transforms produce consistent results."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit(sample_data)

    encoded1 = embedder.transform(sample_data)
    encoded2 = embedder.transform(sample_data)

    np.testing.assert_allclose(encoded1, encoded2, rtol=1e-5)


def test_transform_dtype_conversion(encoder):
    """Test that transform converts input to float32."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit()

    # Create data with different dtype
    data_float64 = np.random.randn(5, 10).astype(np.float64)
    encoded = embedder.transform(data_float64)

    assert encoded.dtype == np.float32


def test_inverse_transform_with_decoder(encoder, decoder, sample_data):
    """Test inverse_transform with decoder available."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder, decoder=decoder)
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)
    decoded = embedder.inverse_transform(encoded)

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (20, 10)
    assert decoded.dtype == np.float32


def test_inverse_transform_without_decoder(encoder, sample_data):
    """Test that inverse_transform raises error when decoder is None."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder, decoder=None)
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)

    with pytest.raises(AttributeError, match="No decoder provided"):
        embedder.inverse_transform(encoded)


def test_inverse_transform_jax_input(encoder, decoder, sample_data):
    """Test inverse_transform with JAX array input."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder, decoder=decoder)
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)
    encoded_jax = jnp.asarray(encoded)
    decoded = embedder.inverse_transform(encoded_jax)

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (20, 10)


def test_to_array_numpy_conversion(encoder):
    """Test _to_array converts numpy arrays to JAX float32."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    jax_array = embedder._to_array(arr)

    assert isinstance(jax_array, jnp.ndarray)
    assert jax_array.dtype == jnp.float32
    assert jax_array.shape == (2, 2)


def test_to_array_jax_conversion(encoder):
    """Test _to_array converts JAX arrays to float32."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)

    jax_input = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
    jax_array = embedder._to_array(jax_input)

    assert isinstance(jax_array, jnp.ndarray)
    assert jax_array.dtype == jnp.float32


def test_repr_without_decoder(encoder):
    """Test __repr__ output without decoder."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    repr_str = repr(embedder)

    assert "SimpleEncoder" in repr_str
    assert "None" in repr_str
    assert "NnxFeatureMapEmbedder" in repr_str


def test_repr_with_decoder(encoder, decoder):
    """Test __repr__ output with decoder."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder, decoder=decoder)
    repr_str = repr(embedder)

    assert "SimpleEncoder" in repr_str
    assert "SimpleDecoder" in repr_str
    assert "NnxFeatureMapEmbedder" in repr_str


def test_encode_decode_cycle_shape(encoder, decoder, sample_data):
    """Test that encode-decode cycle preserves input shape."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder, decoder=decoder)
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)
    decoded = embedder.inverse_transform(encoded)

    assert decoded.shape == sample_data.shape


def test_different_batch_sizes(encoder, sample_data):
    """Test transform works with different batch sizes."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit(sample_data)

    for batch_size in [1, 5, 20, 50]:
        batch_data = np.random.randn(batch_size, 10).astype(np.float32)
        encoded = embedder.transform(batch_data)
        assert encoded.shape == (batch_size, 5)


def test_transform_single_sample(encoder):
    """Test transform with a single sample."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit()

    single_sample = np.random.randn(1, 10).astype(np.float32)
    encoded = embedder.transform(single_sample)

    assert encoded.shape == (1, 5)


def test_jit_compilation_called(encoder, sample_data):
    """Test that JIT-compiled functions are created."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit()

    # Verify jitted functions exist and are callable
    assert callable(embedder.jitted_encoder_call)
    encoded = embedder.transform(sample_data)
    assert encoded.shape == (20, 5)


def test_jit_compilation_decoder(encoder, decoder, sample_data):
    """Test that JIT-compiled decoder is created."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder, decoder=decoder)
    embedder.fit()

    assert callable(embedder.jitted_decoder_call)
    encoded = embedder.transform(sample_data)
    decoded = embedder.inverse_transform(encoded)
    assert decoded.shape == (20, 10)


def test_numerical_values_reasonable(encoder, sample_data):
    """Test that output values are in reasonable range."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit()

    encoded = embedder.transform(sample_data)

    # Check no NaNs or Infs
    assert not np.isnan(encoded).any()
    assert not np.isinf(encoded).any()


def test_transform_preserves_data_independence(encoder):
    """Test that transforming different data produces different results."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit()

    data1 = np.random.randn(10, 10).astype(np.float32)
    data2 = np.random.randn(10, 10).astype(np.float32)

    encoded1 = embedder.transform(data1)
    encoded2 = embedder.transform(data2)

    # Results should be different for different inputs
    assert not np.allclose(encoded1, encoded2)


def test_large_batch(encoder):
    """Test transform with large batch size."""
    embedder = NnxFeatureMapEmbedder(encoder=encoder)
    embedder.fit()

    large_batch = np.random.randn(1000, 10).astype(np.float32)
    encoded = embedder.transform(large_batch)

    assert encoded.shape == (1000, 5)
    assert encoded.dtype == np.float32
