from kooplearn._utils import check_torch_deps

check_torch_deps()
from kooplearn.torch.utils import FeatureMapEmbedder as FeatureMapEmbedder  # noqa: E402

__all__ = ["FeatureMapEmbedder"]
