.. _data_api:

Data
========================

In this page we document functions and classes related to the data-manipulation in Kooplearn. As explained in :ref:`kooplearn's data paradigm <kooplearn_data_paradigm>`, Kooplearn uses a context-based approach to represent dynamical systems data. A context window is just a (usually short) sequence of consecutive observations of the system, enclosing the 'past' in its *lookback window* and the 'future' in its *lookforward window*. Intuitively, everything in the lookback window is the information we need to provide, at inference time, to predict what is in the lookforward window. By using context windows we depart from the usual paradigm in supervised learning in which data is categorized into inputs and outputs. Rather, when studying dynamical system we find it more natural to conceive a "data point" as a context window containing the dynamical information at a given point in time. 

.. autofunction:: kooplearn.data.traj_to_contexts

.. currentmodule:: kooplearn.nn

.. autoclass:: kooplearn.nn.data.ContextsDataset

.. autofunction:: kooplearn.nn.data.traj_to_contexts_dataset



