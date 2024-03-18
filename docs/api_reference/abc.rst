Abstract Base Classes
=====================================

Introduction
------------


Abstract base classes (ABCs) act as templates for other classes, specifying which methods subclasses must implement without defining the implementation itself. ABCs serve as a blueprint, guaranteeing that :code:`kooplearn`'s subclasses adhere to a certain behavior while allowing flexibility in implementation details, ultimately facilitating easier understanding and maintenance of code.

.. module:: kooplearn.abc

Model ABCs
----------

Base Model
~~~~~~~~~~~

Any subclass of the base model implements a different strategy to learn :ref:`Koopman and Transfer operators <primer>` from data. Refer to our :ref:`model zoo <model_zoo>` for a complete list of models already implemented in :code:`kooplearn`.

.. autoclass:: kooplearn.abc.BaseModel
    :members:
    :undoc-members:
    :show-inheritance:


Feature Maps
~~~~~~~~~~~~

.. autoclass:: kooplearn.abc.FeatureMap
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: kooplearn.abc.TrainableFeatureMap
    :members:
    :undoc-members:
    :show-inheritance:

Data & Context Windows
----------------------

As explained in :ref:`the guide on Kooplearn's data paradigm <kooplearn_data_paradigm>`, Kooplearn uses a data paradigm based on context windows. The following base classes show the implementation of context windows expected by Kooplearn, in particular the definition of the basic properties such as :code:`context_length` and methods :code:`lookback` and :code:`lookforward`.

Abstract Context Windows
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: kooplearn.abc.ContextWindow
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: kooplearn.abc.ContextWindowDataset
    :members:
    :undoc-members:
    :show-inheritance:

Tensor Context Windows and :mod:`kooplearn.data`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of the time, an element of the context window is just a Numpy or Torch tensor. In the submodule :mod:`kooplearn.data`, we implemented the functionality to support tensorial context window. 

.. autoclass:: kooplearn.data.TensorContextDataset
    :noindex:
    :members:
    :undoc-members:
    :show-inheritance: