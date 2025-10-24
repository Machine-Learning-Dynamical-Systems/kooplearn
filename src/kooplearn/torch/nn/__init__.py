from kooplearn._utils import check_torch_deps

check_torch_deps()

from kooplearn.torch.nn._base import (
    DynamicAELoss,
    KLContrastiveLoss,
    L2ContrastiveLoss,
    VampLoss,
)

__all__ = [
    "DynamicAELoss",
    "KLContrastiveLoss",
    "L2ContrastiveLoss",
    "VampLoss",
]
