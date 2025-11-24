import pytest
import torch
import torch.nn as nn

from kooplearn.torch.nn._base import (
    AutoEncoderLoss,
    SpectralContrastiveLoss,
    VampLoss,
)


@pytest.fixture
def sample_features():
    """Create sample feature tensors."""
    torch.manual_seed(42)
    batch_size = 16
    feature_dim = 10
    x = torch.randn(batch_size, feature_dim)
    y = torch.randn(batch_size, feature_dim)
    return x, y


@pytest.fixture
def large_batch_features():
    """Create larger sample feature tensors."""
    torch.manual_seed(42)
    batch_size = 128
    feature_dim = 50
    x = torch.randn(batch_size, feature_dim)
    y = torch.randn(batch_size, feature_dim)
    return x, y


@pytest.fixture
def single_sample_features():
    """Create single sample feature tensors."""
    torch.manual_seed(42)
    x = torch.randn(1, 10)
    y = torch.randn(1, 10)
    return x, y


class TestVampLoss:
    """Test suite for VampLoss."""

    def test_init_default_params(self):
        """Test VampLoss initialization with default parameters."""
        loss_fn = VampLoss()
        assert loss_fn.schatten_norm == 2
        assert loss_fn.center_covariances is True

    def test_init_custom_params(self):
        """Test VampLoss initialization with custom parameters."""
        loss_fn = VampLoss(schatten_norm=1, center_covariances=False)
        assert loss_fn.schatten_norm == 1
        assert loss_fn.center_covariances is False

    def test_forward_returns_scalar(self, sample_features):
        """Test that forward pass returns a scalar tensor."""
        x, y = sample_features
        loss_fn = VampLoss()
        loss = loss_fn(x, y)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert isinstance(loss, torch.Tensor)

    def test_forward_dtype(self, sample_features):
        """Test that loss maintains dtype."""
        x, y = sample_features
        loss_fn = VampLoss()
        loss = loss_fn(x, y)

        assert loss.dtype == x.dtype

    def test_forward_schatten_norm_1(self, sample_features):
        """Test VampLoss with Schatten norm p=1."""
        x, y = sample_features
        loss_fn = VampLoss(schatten_norm=1)
        loss = loss_fn(x, y)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_schatten_norm_2(self, sample_features):
        """Test VampLoss with Schatten norm p=2."""
        x, y = sample_features
        loss_fn = VampLoss(schatten_norm=2)
        loss = loss_fn(x, y)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_centered_covariances(self, sample_features):
        """Test VampLoss with centered covariances."""
        x, y = sample_features
        loss_fn = VampLoss(center_covariances=True)
        loss = loss_fn(x, y)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_uncentered_covariances(self, sample_features):
        """Test VampLoss with uncentered covariances."""
        x, y = sample_features
        loss_fn = VampLoss(center_covariances=False)
        loss = loss_fn(x, y)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_symmetric_inputs(self, sample_features):
        """Test VampLoss with identical inputs returns same result."""
        x, _ = sample_features
        loss_fn = VampLoss()
        loss = loss_fn(x, x)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, sample_features):
        """Test that gradients flow through VampLoss."""
        x, y = sample_features
        x.requires_grad_(True)
        y.requires_grad_(True)

        loss_fn = VampLoss()
        loss = loss_fn(x, y)
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(y.grad).any()

    def test_different_batch_sizes(self):
        """Test VampLoss with different batch sizes."""
        loss_fn = VampLoss()

        for batch_size in [1, 2, 8, 32]:
            x = torch.randn(batch_size, 10)
            y = torch.randn(batch_size, 10)
            loss = loss_fn(x, y)

            assert loss.dim() == 0
            assert not torch.isnan(loss)

    def test_large_batch(self, large_batch_features):
        """Test VampLoss with large batch size."""
        x, y = large_batch_features
        loss_fn = VampLoss()
        loss = loss_fn(x, y)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_single_sample(self, single_sample_features):
        """Test VampLoss with single sample."""
        x, y = single_sample_features
        loss_fn = VampLoss()
        loss = loss_fn(x, y)

        assert loss.dim() == 0
        assert not torch.isnan(loss)


class TestSpectralContrastiveLoss:
    """Test suite for SpectralContrastiveLoss."""

    def test_init(self):
        """Test SpectralContrastiveLoss initialization."""
        loss_fn = SpectralContrastiveLoss()
        assert isinstance(loss_fn, nn.Module)

    def test_forward_returns_scalar(self, sample_features):
        """Test that forward pass returns a scalar tensor."""
        x, y = sample_features
        loss_fn = SpectralContrastiveLoss()
        loss = loss_fn(x, y)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert isinstance(loss, torch.Tensor)

    def test_forward_dtype(self, sample_features):
        """Test that loss maintains dtype."""
        x, y = sample_features
        loss_fn = SpectralContrastiveLoss()
        loss = loss_fn(x, y)

        assert loss.dtype == x.dtype

    def test_forward_no_nan(self, sample_features):
        """Test that loss does not contain NaN."""
        x, y = sample_features
        loss_fn = SpectralContrastiveLoss()
        loss = loss_fn(x, y)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_symmetric_inputs(self, sample_features):
        """Test SpectralContrastiveLoss with identical inputs."""
        x, _ = sample_features
        loss_fn = SpectralContrastiveLoss()
        loss = loss_fn(x, x)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, sample_features):
        """Test that gradients flow through SpectralContrastiveLoss."""
        x, y = sample_features
        x.requires_grad_(True)
        y.requires_grad_(True)

        loss_fn = SpectralContrastiveLoss()
        loss = loss_fn(x, y)
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(y.grad).any()

    def test_different_batch_sizes(self):
        """Test SpectralContrastiveLoss with different batch sizes."""
        loss_fn = SpectralContrastiveLoss()

        for batch_size in [2, 4, 8, 32]:
            x = torch.randn(batch_size, 10)
            y = torch.randn(batch_size, 10)
            loss = loss_fn(x, y)

            assert loss.dim() == 0
            assert not torch.isnan(loss)

    def test_large_batch(self, large_batch_features):
        """Test SpectralContrastiveLoss with large batch size."""
        x, y = large_batch_features
        loss_fn = SpectralContrastiveLoss()
        loss = loss_fn(x, y)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_orthogonal_features(self):
        """Test SpectralContrastiveLoss with orthogonal features."""
        batch_size = 16
        feat_dim = 10

        # Create orthogonal features
        q, _ = torch.linalg.qr(torch.randn(feat_dim, feat_dim))
        x = torch.randn(batch_size, 1) @ q[:1, :]
        y = torch.randn(batch_size, 1) @ q[1:2, :]

        loss_fn = SpectralContrastiveLoss()
        loss = loss_fn(x, y)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_different_feature_dims(self):
        """Test SpectralContrastiveLoss with different feature dimensions."""
        loss_fn = SpectralContrastiveLoss()

        for feat_dim in [5, 10, 50, 100]:
            x = torch.randn(16, feat_dim)
            y = torch.randn(16, feat_dim)
            loss = loss_fn(x, y)

            assert loss.dim() == 0
            assert not torch.isnan(loss)


class TestAutoEncoderLoss:
    """Test suite for AutoEncoderLoss."""

    def test_init_default_params(self):
        """Test AutoEncoderLoss initialization with default parameters."""
        loss_fn = AutoEncoderLoss()
        assert loss_fn.alpha_rec == 1.0
        assert loss_fn.alpha_lin == 1.0
        assert loss_fn.alpha_pred == 1.0

    def test_init_custom_params(self):
        """Test AutoEncoderLoss initialization with custom parameters."""
        loss_fn = AutoEncoderLoss(alpha_rec=0.5, alpha_lin=2.0, alpha_pred=1.5)
        assert loss_fn.alpha_rec == 0.5
        assert loss_fn.alpha_lin == 2.0
        assert loss_fn.alpha_pred == 1.5

    def test_forward_returns_scalar(self, sample_features):
        """Test that forward pass returns a scalar tensor."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x + 0.01 * torch.randn_like(x)
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y + 0.01 * torch.randn_like(y)

        loss_fn = AutoEncoderLoss()
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert isinstance(loss, torch.Tensor)

    def test_forward_dtype(self, sample_features):
        """Test that loss maintains dtype."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x.clone()
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y.clone()

        loss_fn = AutoEncoderLoss()
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        assert loss.dtype == x.dtype

    def test_reconstruction_only(self, sample_features):
        """Test AutoEncoderLoss with reconstruction weight only."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x + 0.1 * torch.randn_like(x)
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y.clone()

        loss_fn = AutoEncoderLoss(alpha_rec=1.0, alpha_lin=0.0, alpha_pred=0.0)
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_linearity_only(self, sample_features):
        """Test AutoEncoderLoss with linearity weight only."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x.clone()
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y.clone()

        loss_fn = AutoEncoderLoss(alpha_rec=0.0, alpha_lin=1.0, alpha_pred=0.0)
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_prediction_only(self, sample_features):
        """Test AutoEncoderLoss with prediction weight only."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x.clone()
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y + 0.1 * torch.randn_like(y)

        loss_fn = AutoEncoderLoss(alpha_rec=0.0, alpha_lin=0.0, alpha_pred=1.0)
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, sample_features):
        """Test that gradients flow through all components."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x.clone().requires_grad_(True)
        y_enc = torch.randn(batch_size, latent_dim, requires_grad=True)
        x_evo = torch.randn(batch_size, latent_dim, requires_grad=True)
        y_pred = y.clone().requires_grad_(True)

        loss_fn = AutoEncoderLoss()
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)
        loss.backward()

        assert x_rec.grad is not None
        assert y_enc.grad is not None
        assert x_evo.grad is not None
        assert y_pred.grad is not None

    def test_zero_weights(self, sample_features):
        """Test AutoEncoderLoss with all zero weights."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x.clone()
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y.clone()

        loss_fn = AutoEncoderLoss(alpha_rec=0.0, alpha_lin=0.0, alpha_pred=0.0)
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        assert loss == 0.0

    def test_large_weights(self, sample_features):
        """Test AutoEncoderLoss with large weights."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x + 0.1 * torch.randn_like(x)
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y + 0.1 * torch.randn_like(y)

        loss_fn = AutoEncoderLoss(alpha_rec=100.0, alpha_lin=100.0, alpha_pred=100.0)
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_perfect_reconstruction(self, sample_features):
        """Test AutoEncoderLoss with perfect reconstructions."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x.clone()
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = y_enc.clone()  # Make evolved latent match encoded output
        y_pred = y.clone()

        loss_fn = AutoEncoderLoss()
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        # Should be zero since all three terms are perfectly satisfied
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_different_batch_sizes(self):
        """Test AutoEncoderLoss with different batch sizes."""
        loss_fn = AutoEncoderLoss()

        for batch_size in [1, 2, 8, 32]:
            x = torch.randn(batch_size, 10)
            y = torch.randn(batch_size, 10)
            x_rec = x.clone()
            y_enc = torch.randn(batch_size, 5)
            x_evo = torch.randn(batch_size, 5)
            y_pred = y.clone()

            loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

            assert loss.dim() == 0
            assert not torch.isnan(loss)

    def test_large_batch(self, large_batch_features):
        """Test AutoEncoderLoss with large batch size."""
        x, y = large_batch_features
        batch_size, _feat_dim = x.shape
        latent_dim = 10

        x_rec = x + 0.01 * torch.randn_like(x)
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y + 0.01 * torch.randn_like(y)

        loss_fn = AutoEncoderLoss()
        loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_weighted_combination(self, sample_features):
        """Test that weights correctly scale loss components."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape
        latent_dim = 5

        x_rec = x + torch.randn_like(x)
        y_enc = torch.randn(batch_size, latent_dim)
        x_evo = torch.randn(batch_size, latent_dim)
        y_pred = y + torch.randn_like(y)

        # Compute with different weight configurations
        loss_fn1 = AutoEncoderLoss(alpha_rec=1.0, alpha_lin=0.0, alpha_pred=0.0)
        loss_fn2 = AutoEncoderLoss(alpha_rec=2.0, alpha_lin=0.0, alpha_pred=0.0)

        loss1 = loss_fn1(x, y, x_rec, y_enc, x_evo, y_pred)
        loss2 = loss_fn2(x, y, x_rec, y_enc, x_evo, y_pred)

        # Loss should roughly double when weight doubles
        assert torch.isclose(loss2, 2 * loss1, rtol=0.01)


class TestLossesIntegration:
    """Integration tests for all loss functions."""

    def test_losses_as_modules(self):
        """Test that all losses are proper nn.Module instances."""
        losses = [
            VampLoss(),
            SpectralContrastiveLoss(),
            AutoEncoderLoss(),
        ]

        for loss_fn in losses:
            assert isinstance(loss_fn, nn.Module)

    def test_losses_in_training_loop(self, sample_features):
        """Test losses in a simple training loop."""
        x, y = sample_features
        batch_size, _feat_dim = x.shape

        losses = {
            "vamp": VampLoss(),
            "spectral": SpectralContrastiveLoss(),
            "ae": AutoEncoderLoss(),
        }

        # Simulate one training step
        for name, loss_fn in losses.items():
            if name == "ae":
                x_rec = x.clone()
                y_enc = torch.randn(batch_size, 5)
                x_evo = torch.randn(batch_size, 5)
                y_pred = y.clone()
                loss = loss_fn(x, y, x_rec, y_enc, x_evo, y_pred)
            else:
                loss = loss_fn(x, y)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_losses_device_consistency(self):
        """Test that losses work on CPU."""
        x = torch.randn(16, 10)
        y = torch.randn(16, 10)

        losses = [
            VampLoss(),
            SpectralContrastiveLoss(),
        ]

        for loss_fn in losses:
            loss_fn.to("cpu")
            x_cpu = x.to("cpu")
            y_cpu = y.to("cpu")
            loss = loss_fn(x_cpu, y_cpu)

            assert not torch.isnan(loss)
