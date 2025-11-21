import os
from typing import Optional, Union

import numpy as np
from sklearn.datasets import fetch_openml


def fetch_ordered_mnist(
    *,
    num_digits: int = 10,
    data_home: Optional[Union[str, os.PathLike]] = None,
    cache: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
):
    """
    Fetch the MNIST dataset and return an *ordered* subset interleaving samples
    from each digit class.

    This function wraps :func:`sklearn.datasets.fetch_openml` for the MNIST
    dataset (OpenML ID 554) and reorders the samples so that digits ``0`` through
    ``num_digits - 1`` are **interleaved** in the output. This is useful for
    generating class-balanced or periodic sequences for Koopman operator
    regression experiments.

    The MNIST dataset contains 70,000 grayscale handwritten digits
    (60,000 for training and 10,000 for testing) of size 28×28.

    Parameters
    ----------
    num_digits : int, default=10
        Number of digit classes to include, from 1 to 10.
        For example, ``num_digits=3`` returns only digits ``0``, ``1``, and ``2``.

    data_home : str or path-like, optional
        Specify an alternative download and cache folder for the dataset.
        By default, scikit-learn stores data in ``~/scikit_learn_data``.

    cache : bool, default=True
        Whether to cache the downloaded dataset.

    n_retries : int, default=3
        Number of times to retry downloading if network errors occur.

    delay : float, default=1.0
        Number of seconds between retries during download.

    Returns
    -------
    images : ndarray of shape (n_samples, 28, 28)
        Array of grayscale MNIST images (uint8).

    targets : ndarray of shape (n_samples,)
        Corresponding digit labels (integers in ``[0, num_digits - 1]``).

    Notes
    -----
    The dataset is reordered so that classes are interleaved in the returned
    arrays. For example, with ``num_digits=3``, the ordering will be:

    ``[0, 1, 2, 0, 1, 2, 0, 1, 2, ...]``

    Examples
    --------
    >>> from kooplearn.datasets import fetch_ordered_mnist
    >>> images, targets = fetch_ordered_mnist(num_digits=3)
    >>> images.shape
    (20709, 28, 28)
    >>> np.unique(targets)
    array([0, 1, 2])
    """
    if not (1 <= num_digits <= 10):
        raise ValueError(f"`num_digits` must be between 1 and 10, got {num_digits}.")

    # Fetch raw MNIST data (OpenML ID 554)
    X, y = fetch_openml(
        data_id=554,
        data_home=data_home,
        cache=cache,
        n_retries=n_retries,
        delay=delay,
        return_X_y=True,
        as_frame=False,
    )

    # Convert to arrays
    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y, dtype=int)

    # Reshape to (n_samples, 28, 28)
    images = X.reshape(-1, 28, 28)
    targets = y

    # Get class indices for digits 0..num_digits-1
    digits_indexes = {
        digit: np.flatnonzero(targets == digit) for digit in range(num_digits)
    }

    # Compute interleaved ordering:
    # e.g. [0_0, 1_0, 2_0, 0_1, 1_1, 2_1, ...]
    def interleave(indices_dict, classes):
        sizes = [len(indices_dict[d]) for d in classes]
        n_min = min(sizes)
        n_classes = len(classes)
        dtype = indices_dict[classes[0]].dtype

        # Allocate array for interleaved indices
        ordered = np.empty(n_min * n_classes, dtype=dtype)
        for start, cls in enumerate(classes):
            ordered[start::n_classes] = indices_dict[cls][:n_min]
        return ordered

    ordering_perm = interleave(digits_indexes, list(range(num_digits)))

    # Apply the new ordering
    images = images[ordering_perm]
    targets = targets[ordering_perm]

    # Scaling data
    images = (images / 255.0).astype("float64")

    return images, targets
