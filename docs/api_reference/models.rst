Models
=====================================

.. currentmodule:: kooplearn.models

Linear & Kernel models
----------------------

.. autoclass:: ExtendedDMD
    :members:

.. autoclass:: KernelDMD
    :members:

.. autoclass:: DMD

Deep-Learning models
--------------------
Kooplearn's deep learning components are implemented using `pytorch lightning <https://lightning.ai>`_ as it allows to easily set up complex machine learning pipelines in a modular way, and it comes with an extensive toolset out of the box. Broadly speaking, Kooplearn's implementation expect you to provide:

1. One or more :code:`torch.nn.Module` objects defining the neural-network, along with the arguments passed at initialization.
2. A torch optimizer along with the arguments passed at initialization.
3. An `initialized` :code:`lightning.Trainer` object, in which the user can define loggers, callbacks, scheduler etc.

Kooplearn, in turn will handle the creation of a :code:`lightning.LightningModule` internally. Models are then fitted by calling the :code:`fit` method, which has roughly the same signature of :code:`lightning.Trainer().fit`, and accepts both `torch dataloaders <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_ and `lightning datamodules <https://lightning.ai/docs/pytorch/stable/data/datamodule.html>`_.

.. autoclass:: DeepEDMD

.. autoclass:: DynamicAE
    :members:

.. autoclass:: ConsistentAE
    :members:
    
.. footbibliography::
