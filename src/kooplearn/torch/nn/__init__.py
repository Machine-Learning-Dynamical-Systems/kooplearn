from kooplearn.utils import check_torch_deps

check_torch_deps()

from kooplearn.torch.nn._base import (
    VampLoss, 
    L2ContrastiveLoss, 
    KLContrastiveLoss
    )

__all__ = [
    "VampLoss",
    "L2ContrastiveLoss",
    "KLContrastiveLoss",
    ]
