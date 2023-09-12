Feature Maps
=====================================
Feature maps are functions mapping inputs to a feature space. The abstract implementation of Kooplearn's feature maps can be found in :class:`kooplearn.abc.FeatureMap`. Theu should be callable objects accepting batches of data points of shape ``(batch_size, *input_dims)`` and returning batches of feature vectors of shape ``(batch_size, *output_dims)``. 

Feature maps can also be learned directly from data in an unsupervised way. Kooplearn also exposes the abstract class :class:`kooplearn.abc.TrainableFeatureMap` for this purpose. It is a subclass of :class:`kooplearn.abc.FeatureMap` and adds the method :meth:`kooplearn.abc.TrainableFeatureMap.fit`. 


Implemented Feature Maps:
-------------------------

.. currentmodule:: kooplearn.models.feature_maps

.. autoclass:: DPNet
    :members:

.. autoclass:: IdentityFeatureMap

.. autoclass:: ConcatenateFeatureMaps

.. footbibliography::