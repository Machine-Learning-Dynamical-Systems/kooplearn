from kooplearn.models.feature_maps.base import (
    ConcatenateFeatureMaps,
    IdentityFeatureMap,
)

try:
    from kooplearn._src.check_deps import check_torch_deps

    check_torch_deps()
    from kooplearn.models.feature_maps.dpnets import DPNet
    from kooplearn.models.feature_maps.vampnets import VAMPNet
except ImportError:
    pass
