import pytest

from kooplearn.datasets import fetch_ordered_mnist


def test_fetch_ordered_mnist_invalid_digits():
    """Ensure invalid num_digits raises ValueError."""
    with pytest.raises(ValueError, match="between 1 and 10"):
        fetch_ordered_mnist(num_digits=0)
    with pytest.raises(ValueError, match="between 1 and 10"):
        fetch_ordered_mnist(num_digits=11)
