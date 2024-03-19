Datasets
=====================================

Introduction
------------

Kooplearn comes bundled with a number of example datasets which can be used to test and benchmark :ref:`Koopman and Transfer operators <primer>` models. The datasets cover both deterministic and stochastic dynamical systems.

.. module:: kooplearn.datasets

Basic Usage
-----------

Every dataset in :mod:`kooplearn.datasets` can be called following the same simple API. This is as easy as 

#. Initialize the dataset
#. Sample a new trajectory of :code:`n_samples` steps from an initial condition :code:`initial_condition` by calling the sampling method :code:`sample(initial_condition, n_samples)`.

For example, 

.. code-block:: python

   from kooplearn.datasets import LogisticMap

    # Defining the number of samples for each data split
    train_samples = 50000 
    test_samples = 1000

    logmap = LogisticMap(N = 20, rng_seed = 0) # Setting the rng_seed for reproducibility

    initial_condition = 0.5 # Setting the initial condition x_0 to start sampling the map

    datasets = {
        'train': logmap.sample(initial_condition, train_samples),
        'test': logmap.sample(initial_condition, test_samples)
    }

Some toy datasets such as :class:`kooplearn.datasets.LogisticMap` and :class:`kooplearn.datasets.LinearModel` also exposes an :code:`eig` method returning ground truth spectral decomposition of the Koopman / Transfer Operator.

Datasets
--------

Generic Dataset Classes
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: kooplearn.datasets.misc.DiscreteTimeDynamics
    :members:
    :undoc-members:

Stochastic Dynamical Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: kooplearn.datasets.LinearModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: kooplearn.datasets.LogisticMap
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: kooplearn.datasets.MullerBrownPotential
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: kooplearn.datasets.LangevinTripleWell1D
    :members:
    :undoc-members:
    :show-inheritance:

Deterministic Dynamical Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: kooplearn.datasets.DuffingOscillator
    :members:

.. autoclass:: kooplearn.datasets.Lorenz63
    :members:

.. footbibliography::