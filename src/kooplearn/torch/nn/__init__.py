from kooplearn._utils import check_torch_deps

check_torch_deps()

from kooplearn.torch.nn._base import (
    AutoEncoderLoss,
    KLContrastiveLoss,
    SpectralContrastiveLoss,
    VampLoss,
)

__all__ = [
    "AutoEncoderLoss",
    "SpectralContrastiveLoss",
    "KLContrastiveLoss",
    "VampLoss",
]
