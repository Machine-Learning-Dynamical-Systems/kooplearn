
.. _api_entrypoint:

API Reference
=======================

Least-Squares estimators
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: kooplearn.linear_model
   :no-members:
   :no-inherited-members:

.. currentmodule:: kooplearn.linear_model

.. autosummary::
   :toctree: ../generated/
   :template: class.rst

   Ridge

Kernel-based estimators
^^^^^^^^^^^^^^^^^^^^^^^   

.. automodule:: kooplearn.kernel
   :no-members:
   :no-inherited-members:

.. currentmodule:: kooplearn.kernel

.. autosummary::
   :toctree: ../generated/
   :template: class.rst

   KernelRidge
   NystroemKernelRidge
   GeneratorDirichlet

Preprocessing utilities
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: kooplearn.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: kooplearn.preprocessing

.. autosummary::
   :toctree: ../generated/
   :template: class.rst

   TimeDelayEmbedding
   FeatureFlattener

Datasets
^^^^^^^^

.. automodule:: kooplearn.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: kooplearn.datasets

.. autosummary::
   :toctree: ../generated/

   compute_prinz_potential_eig
   fetch_ordered_mnist
   make_duffing
   make_linear_system
   make_logistic_map
   make_lorenz63
   make_prinz_potential
   make_regime_switching_var

PyTorch Integration
^^^^^^^^^^^^^^^^^^^

.. automodule:: kooplearn.torch.nn
   :no-members:
   :no-inherited-members:

.. currentmodule:: kooplearn.torch.nn

.. autosummary::
   :toctree: ../generated/
   :template: class.rst

   SpectralContrastiveLoss
   VampLoss
   AutoEncoderLoss

JAX Integration
^^^^^^^^^^^^^^^

.. automodule:: kooplearn.jax.nn
   :no-members:
   :no-inherited-members:

.. currentmodule:: kooplearn.jax.nn

.. autosummary::
   :toctree: ../generated/
   :template: class.rst

   spectral_contrastive_loss
   vamp_loss
   autoencoder_loss