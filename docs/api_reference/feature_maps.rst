Feature Maps
============

Introduction
------------
Feature maps are functions :math:`\phi` which "featurize" an input :math:`x` and lift it onto a feature space :math:`\phi(x)`. The abstract implementation of Kooplearn's feature maps can be found in :class:`kooplearn.abc.FeatureMap`. They should be callable objects accepting batches of data points of shape ``(batch_size, *input_dims)`` and returning batches of feature vectors of shape ``(batch_size, *output_dims)``. 

Feature maps can also be learned directly from data. To this end, Kooplearn exposes the abstract class :class:`kooplearn.abc.TrainableFeatureMap` as a subclass of :class:`kooplearn.abc.FeatureMap` with the added method :meth:`kooplearn.abc.TrainableFeatureMap.fit`. 


.. currentmodule:: kooplearn.models.feature_maps

Feature Maps
------------

Basic Feature Maps
~~~~~~~~~~~~~~~~~~

.. autoclass:: IdentityFeatureMap

.. autoclass:: ConcatenateFeatureMaps

Deep-Learning Feature Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~
Kooplearn's deep learning components are implemented using `pytorch lightning <https://lightning.ai>`_ as it allows to easily set up complex machine learning pipelines in a modular way, and it comes with an extensive toolset out of the box. Broadly speaking, Kooplearn's implementation expect you to provide:

1. One or more :code:`torch.nn.Module` objects defining the neural-network, along with the arguments passed at initialization.
2. A torch optimizer along with the arguments passed at initialization.
3. An `initialized` :code:`lightning.Trainer` object, in which the user can define loggers, callbacks, scheduler etc.

Kooplearn, in turn will handle the creation of a :code:`lightning.LightningModule` internally. Models are then fitted by calling the :code:`fit` method, which has roughly the same signature of :code:`lightning.Trainer().fit`, and accepts both `torch dataloaders <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ and `lightning datamodules <https://lightning.ai/docs/pytorch/stable/data/datamodule.html>`_.

 

.. autoclass:: NNFeatureMap
    :members:

.. footbibliography::