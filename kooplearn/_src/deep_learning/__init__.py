try:
    from kooplearn._src.deep_learning.architectures import MLPModel
    from kooplearn._src.deep_learning.data_utils import TimeseriesDataModule, TimeseriesDataset
    from kooplearn._src.deep_learning.feature_maps import DPNetFeatureMap
    from kooplearn._src.deep_learning.models import BruntonModel
except ImportError as e:
    raise ImportError("deepkoopman must be installed to use the deep learning API. You can install it with"
                      " `pip install kooplearn[deepkoopman]`."
                      "Original error message: {}".format(e))
