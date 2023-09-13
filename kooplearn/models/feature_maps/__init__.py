from kooplearn.models.feature_maps.base import IdentityFeatureMap, ConcatenateFeatureMaps
try:
    from kooplearn.models.feature_maps.DPNets import DPNet
except ImportError:
    pass
