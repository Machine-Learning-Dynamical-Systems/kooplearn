from kooplearn.models.feature_maps.base import IdentityFeatureMap, ConcatenateFeatureMaps
try:
    from kooplearn._src.check_deps import check_torch_deps
    check_torch_deps()
    from kooplearn.models.feature_maps.vampnets import VAMPNet
    from kooplearn.models.feature_maps.dpnets import DPNet
except ImportError:
    pass
