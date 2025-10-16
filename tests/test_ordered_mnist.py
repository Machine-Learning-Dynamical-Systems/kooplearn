import numpy as np
import pytest

from kooplearn.datasets import fetch_ordered_mnist


def test_fetch_ordered_mnist_basic(monkeypatch):
    """Test that the function returns correctly shaped arrays and interleaves properly."""

    # Mock fetch_openml to avoid network calls
    n_classes = 3
    n_samples_per_class = 5
    n_total = n_classes * n_samples_per_class

    fake_X = np.arange(n_total * 28 * 28).reshape(n_total, 28 * 28)
    fake_y = np.repeat(np.arange(n_classes), n_samples_per_class)

    def fake_fetch_openml(*args, **kwargs):
        return fake_X, fake_y

    monkeypatch.setattr(
        "kooplearn.datasets.fetch_ordered_mnist.fetch_openml", fake_fetch_openml
    )

    images, targets = fetch_ordered_mnist(num_digits=3)

    # Shapes
    assert images.shape == (n_total, 28, 28)
    assert targets.shape == (n_total,)

    # All classes included
    assert set(targets) == {0, 1, 2}

    # Check interleaving pattern (first few entries should cycle 0,1,2)
    pattern = targets[:6].tolist()
    assert pattern == [0, 1, 2, 0, 1, 2]


def test_fetch_ordered_mnist_invalid_digits():
    """Ensure invalid num_digits raises ValueError."""
    with pytest.raises(ValueError, match="between 1 and 10"):
        fetch_ordered_mnist(num_digits=0)
    with pytest.raises(ValueError, match="between 1 and 10"):
        fetch_ordered_mnist(num_digits=11)
