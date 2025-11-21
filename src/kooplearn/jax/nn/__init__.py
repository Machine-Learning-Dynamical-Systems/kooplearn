from kooplearn._utils import check_jax_deps

check_jax_deps()

from kooplearn.jax.nn._functional import (  # noqa: E402
    autoencoder_loss,
    spectral_contrastive_loss,
    vamp_loss,
)

__all__ = [
    "autoencoder_loss",
    "spectral_contrastive_loss",
    "vamp_loss",
]
