from kooplearn._utils import check_jax_deps

check_jax_deps()
from kooplearn.jax.utils import (
    NnxFeatureMapEmbedder as NnxFeatureMapEmbedder,
)

__all__ = ["NnxFeatureMapEmbedder"]
