from kooplearn.models.feature_maps.base import IdentityFeatureMap, ConcatenateFeatureMaps
try:
    from kooplearn.models.feature_maps.DPNets.DPNetsFeatureMap import DPNetsFeatureMap
except ImportError:
    pass
# 