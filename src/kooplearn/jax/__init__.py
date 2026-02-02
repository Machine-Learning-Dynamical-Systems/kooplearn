from kooplearn._utils import check_jax_deps

check_jax_deps()
from kooplearn.jax.utils import (  # noqa: E402
    NnxFeatureMapEmbedder as NnxFeatureMapEmbedder,
)

__all__ = ["NnxFeatureMapEmbedder"]
