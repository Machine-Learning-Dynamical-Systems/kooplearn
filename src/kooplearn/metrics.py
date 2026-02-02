import numpy as np


def directed_hausdorff_distance(pred: np.ndarray, reference: np.ndarray):
    """One-sided directed Hausdorff distance between two 1D sets. Useful for computing distances
    between estimated eigenvalues

    Calculates the directed Hausdorff distance
    :math:`\\vec{H}(A, B) = \\max_{a \\in A} \\min_{b \\in B} \\|a - b\\|_p` where :math:`A` is the
    set of points in ``pred`` and :math:`B` is the set of points in ``reference``. The current
    implementation uses the :math:`L_1` norm: :math:`\\|a - b\\|_1 = |a - b|`.

    Parameters
    ----------
    pred : numpy.ndarray
        The set of predicted points :math:`A`. Must be a 1D array.
    reference : numpy.ndarray
        The set of reference points :math:`B`. Must be a 1D array.

    Returns
    -------
    float
        The directed Hausdorff distance between ``pred``and ``reference``.

    Raises
    ------
    AssertionError
        If ``pred`` or ``reference`` are not 1-dimensional arrays.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from kooplearn.metrics import directed_hausdorff_distance
        pred = np.array([1, 5, 6])
        reference = np.array([2, 4, 7])
        directed_hausdorff_distance(pred, reference)
        # Will print np.float64(1.0)
    """
    pred = np.asanyarray(pred)
    reference = np.asanyarray(reference)
    assert pred.ndim == 1
    assert reference.ndim == 1

    distances = np.zeros((pred.shape[0], reference.shape[0]), dtype=np.float64)
    for pred_idx, pred_pt in enumerate(pred):
        for reference_idx, reference_pt in enumerate(reference):
            distances[pred_idx, reference_idx] = np.abs(pred_pt - reference_pt)
    hausdorff_dist = np.max(np.min(distances, axis=1))
    return hausdorff_dist
