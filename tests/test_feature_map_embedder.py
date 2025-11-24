import numpy as np
import pytest
import torch
import torch.nn as nn

from kooplearn.torch.utils import FeatureMapEmbedder


# Simple test neural networks
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=10, latent_dim=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        return self.linear(x)


class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=5, output_dim=10):
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return self.linear(z)


@pytest.fixture
def encoder():
    return SimpleEncoder(input_dim=10, latent_dim=5)


@pytest.fixture
def decoder():
    return SimpleDecoder(latent_dim=5, output_dim=10)


@pytest.fixture
def sample_data():
    return np.random.randn(20, 10).astype(np.float32)


def test_init_encoder_only(encoder):
    """Test initialization with encoder only."""
    embedder = FeatureMapEmbedder(encoder=encoder)
    assert embedder.encoder is encoder
    assert embedder.decoder is None
    assert embedder.device in ["cpu", "cuda"]


def test_init_encoder_decoder(encoder, decoder):
    """Test initialization with both encoder and decoder."""
    embedder = FeatureMapEmbedder(encoder=encoder, decoder=decoder)
    assert embedder.encoder is encoder
    assert embedder.decoder is decoder
    assert embedder.device in ["cpu", "cuda"]


def test_init_explicit_device(encoder):
    """Test initialization with explicit device specification."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")
    assert embedder.device == "cpu"


def test_fit_returns_self(encoder, sample_data):
    """Test that fit() returns self (sklearn API compliance)."""
    embedder = FeatureMapEmbedder(encoder=encoder)
    result = embedder.fit(sample_data)
    assert result is embedder


def test_fit_with_y(encoder, sample_data):
    """Test that fit() accepts y parameter (sklearn API)."""
    embedder = FeatureMapEmbedder(encoder=encoder)
    y = np.random.randn(20, 5)
    result = embedder.fit(sample_data, y=y)
    assert result is embedder


def test_transform_numpy_input(encoder, sample_data):
    """Test transform with numpy array input."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")
    embedder.fit(sample_data)
    encoded = embedder.transform(sample_data)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (20, 5)
    assert encoded.dtype in [np.float32, np.float64]


def test_transform_tensor_input(encoder, sample_data):
    """Test transform with torch tensor input."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")
    embedder.fit(sample_data)

    tensor_input = torch.from_numpy(sample_data).float()
    encoded = embedder.transform(tensor_input)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (20, 5)


def test_transform_output_consistency(encoder, sample_data):
    """Test that multiple transforms produce consistent results."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")
    embedder.fit(sample_data)

    encoded1 = embedder.transform(sample_data)
    encoded2 = embedder.transform(sample_data)

    np.testing.assert_allclose(encoded1, encoded2, rtol=1e-6)


def test_inverse_transform_with_decoder(encoder, decoder, sample_data):
    """Test inverse_transform with decoder available."""
    embedder = FeatureMapEmbedder(encoder=encoder, decoder=decoder, device="cpu")
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)
    decoded = embedder.inverse_transform(encoded)

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (20, 10)


def test_inverse_transform_without_decoder(encoder, sample_data):
    """Test that inverse_transform raises error when decoder is None."""
    embedder = FeatureMapEmbedder(encoder=encoder, decoder=None, device="cpu")
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)

    with pytest.raises(AttributeError, match="No decoder provided"):
        embedder.inverse_transform(encoded)


def test_inverse_transform_tensor_input(encoder, decoder, sample_data):
    """Test inverse_transform with tensor input."""
    embedder = FeatureMapEmbedder(encoder=encoder, decoder=decoder, device="cpu")
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)
    encoded_tensor = torch.from_numpy(encoded).float()
    decoded = embedder.inverse_transform(encoded_tensor)

    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (20, 10)


def test_to_tensor_numpy_conversion(encoder):
    """Test _to_tensor converts numpy arrays correctly."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    tensor = embedder._to_tensor(arr)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor.shape == (2, 2)


def test_to_tensor_tensor_conversion(encoder):
    """Test _to_tensor converts tensors correctly."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")

    tensor_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    tensor = embedder._to_tensor(tensor_input)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"


def test_repr_without_decoder(encoder):
    """Test __repr__ output without decoder."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")
    repr_str = repr(embedder)

    assert "SimpleEncoder" in repr_str
    assert "None" in repr_str
    assert "cpu" in repr_str


def test_repr_with_decoder(encoder, decoder):
    """Test __repr__ output with decoder."""
    embedder = FeatureMapEmbedder(encoder=encoder, decoder=decoder, device="cpu")
    repr_str = repr(embedder)

    assert "SimpleEncoder" in repr_str
    assert "SimpleDecoder" in repr_str
    assert "cpu" in repr_str


def test_encode_decode_cycle_shape(encoder, decoder, sample_data):
    """Test that encode-decode cycle preserves shape."""
    embedder = FeatureMapEmbedder(encoder=encoder, decoder=decoder, device="cpu")
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)
    decoded = embedder.inverse_transform(encoded)

    assert decoded.shape == sample_data.shape


def test_transform_eval_mode(encoder, sample_data):
    """Test that transform sets encoder to eval mode."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")
    embedder.fit(sample_data)

    encoder.train()  # Set to train mode
    embedder.transform(sample_data)

    # After transform, encoder should be in eval mode
    assert not encoder.training


def test_inverse_transform_eval_mode(encoder, decoder, sample_data):
    """Test that inverse_transform sets decoder to eval mode."""
    embedder = FeatureMapEmbedder(encoder=encoder, decoder=decoder, device="cpu")
    embedder.fit(sample_data)

    encoded = embedder.transform(sample_data)
    decoder.train()  # Set to train mode
    embedder.inverse_transform(encoded)

    # After inverse_transform, decoder should be in eval mode
    assert not decoder.training


def test_different_batch_sizes(encoder, sample_data):
    """Test transform works with different batch sizes."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")
    embedder.fit(sample_data)

    for batch_size in [1, 5, 20, 50]:
        batch_data = np.random.randn(batch_size, 10).astype(np.float32)
        encoded = embedder.transform(batch_data)
        assert encoded.shape == (batch_size, 5)


def test_no_grad_in_transform(encoder, sample_data):
    """Test that transform doesn't create gradient graph."""
    embedder = FeatureMapEmbedder(encoder=encoder, device="cpu")
    embedder.fit(sample_data)

    tensor_input = torch.from_numpy(sample_data).float().requires_grad_(True)
    encoded_np = embedder.transform(tensor_input)

    # Result should not have gradients enabled
    assert not torch.from_numpy(encoded_np).requires_grad
