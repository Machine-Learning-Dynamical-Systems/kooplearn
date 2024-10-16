Neural networks
===============

.. currentmodule:: kooplearn.nn

Main module
-----------

Loss functions
~~~~~~~~~~~~~~~

.. autoclass:: kooplearn.nn.EYMLoss
    :members:

.. autoclass:: kooplearn.nn.DPLoss
    :members:

.. autoclass:: kooplearn.nn.VAMPLoss
    :members:


Functional components
---------------------

Score & loss functions
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kooplearn.nn.functional.eym_score

.. autofunction:: kooplearn.nn.functional.vamp_score

.. autofunction:: kooplearn.nn.functional.deepprojection_score

.. autofunction:: kooplearn.nn.functional.log_fro_metric_deformation_loss


Linear algebra utilities
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: kooplearn.nn.functional.covariance

.. autofunction:: kooplearn.nn.functional.sqrtmh

Bibliography
------------

.. footbibliography::