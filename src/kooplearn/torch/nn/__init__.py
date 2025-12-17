from kooplearn._utils import check_torch_deps

check_torch_deps()

from kooplearn.torch.nn._base import (  # noqa: E402
    AutoEncoderLoss,
    EnergyLoss,
    SpectralContrastiveLoss,
    VampLoss,
)

__all__ = [
    "AutoEncoderLoss",
    "EnergyLoss",
    "SpectralContrastiveLoss",
    "VampLoss",
]
