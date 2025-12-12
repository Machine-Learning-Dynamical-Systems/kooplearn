.. kooplearn documentation master file, created by
   sphinx-quickstart on Wed Oct 15 05:56:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Kooplearn
=========

``kooplearn`` is a Python library to learn :ref:`evolution operators <primer>` —  also known as Koopman :cite:p:`p1-Koopman1931` or Transfer :cite:p:`p1-Applebaum2009` operators —  from data. ``kooplearn`` models can

1. Predict the evolution of states *and* observables.
2. Estimate the eigenvalues and eigenfucntions of the learned evolution operators.
3. Compute the `dynamic mode decomposition <https://en.wikipedia.org/wiki/Dynamic_mode_decomposition>`_ of states *and* observables.
4. Learn neural-network representations :math:`x_t \mapsto \varphi(x_t)` for evolution operators.

Why Choosing ``kooplearn``?
^^^^^^^^^^^^^^^^^^^^^^^

1. It is easy to use, and strictly adheres to the `scikit-learn API <https://scikit-learn.org/stable/api/index.html>`_.
2. :ref:`Kernel estimators <api_kernel>` are state-of-the-art: 
   
   * ``kooplearn`` implements the *Reduced Rank Regressor* from :cite:`p1-Kostic2022` which is provably better :cite:`p1-Kostic2023SpectralRates` than the classical kernel DMD :cite:`p1-Williams2015_KDMD` in estimating eigenvalues and eigenfunctions. 
   * It also implements Nyström estimators :cite:`p1-Meanti2023` and randomized estimators :cite:`p1-turri2023randomized` for :doc:`blazingly fast <examples/ala2_nys_tutorial.ipynb>` kernel learning.

3. Includes representation learning losses (implemented both in :ref:`PyTorch <api_torchnn>` and :ref:`JAX <api_jaxnn>`) to train neural-network Koopman embeddings.
4. Offers a collection of :ref:`datasets <api_datasets>` for benchmarking evolution operator learning algorithms.

Installation
^^^^^^^^^^^^

To install the core version of ``kooplearn``, run

.. tab-set::
    :class: outline

    .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            pip install kooplearn

    .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            uv add kooplearn

To be able to learn neural-network representations using the representation-learning losses in ``kooplearn.torch`` or ``kooplearn.jax``, run

.. tab-set::
    :class: outline

    .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash
            
            # Torch
            pip install "kooplearn[torch]" 
            # JAX
            pip install "kooplearn[jax]"

    .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            # Torch
            uv add "kooplearn[torch]"
            # JAX
            uv add "kooplearn[jax]"

Quickstart
^^^^^^^^^^

.. code-block:: python 

   import numpy as np
   from kooplearn.datasets import make_duffing
   from kooplearn.kernel import KernelRidge

   # Sample data from the Duffing oscillator
   data = make_duffing(X0 = np.array([0, 0]), n_steps=1000)

   # Fit the model
   model = KernelRidge(n_components=4, kernel='rbf', alpha=1e-5)
   model.fit(data)

Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :child-align: center
      :columns: 6
      
      .. card:: :iconify:`material-symbols:rocket-launch width=2em height=2em` Getting started
         :class-card: sd-text-black sd-bg-light sd-text-center sd-align-middle
         :link: examples/linear_system.html

   .. grid-item::
      :columns: 6

      .. card:: :material-regular:`library_books;2em` Theory Primer
         :class-card: sd-text-black sd-bg-light sd-text-center sd-align-middle
         :link: primer.html

   .. grid-item::
      :columns: 6

      .. card:: :material-regular:`science;2em` Examples
         :class-card: sd-text-black sd-bg-light sd-text-center sd-align-middle
         :link: examples.html


   .. grid-item::
      :columns: 6

      .. card:: :material-regular:`menu_book;2em` API reference
         :class-card: sd-text-black sd-bg-light sd-text-center sd-align-middle
         :link: api/index.html

----

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   Getting Started <examples/linear_system.md>
   A Primer on Evolution Operators <primer.md>
   The Operator Way <https://pietronvll.github.io/the-operator-way.html>

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   Eigenfunctions of a 1D Potential <examples/prinz_potential.ipynb>
   Dynamic Mode Decomposition <examples/fluid_flow_dmd.ipynb>
   Using the infinitesimal generator of overdamped Langevin dynamics <examples/prinz_potential_dirichlet generator>
   Fast Kernel Models <examples/kernel_methods.ipynb>
   Analysing Molecular Dynamics <examples/ala2_nys_tutorial.ipynb>
   Switching System <examples/switching_system.ipynb>
   Ordered MNIST (torch) <examples/ordered_mnist_torch.ipynb>
   Ordered MNIST (jax) <examples/ordered_mnist_jax.ipynb>
   Logistic Map (torch) <examples/logistic_map.ipynb>
   

.. toctree::
   :maxdepth: 4
   :caption: API
   :hidden:

   api/main
   api/linear_model
   api/kernel
   api/preprocessing
   api/metrics
   api/datasets
   api/torch
   api/jax
   
.. bibliography::
   :keyprefix: p1-
   :style: unsrt