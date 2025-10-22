.. kooplearn documentation master file, created by
   sphinx-quickstart on Wed Oct 15 05:56:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Kooplearn
=========

Kooplearn is a Python library to learn :ref:`evolution operators <primer>` —  also known as Koopman :cite:p:`Koopman1931` or Transfer :cite:p:`Applebaum2009` operators —  from data. ``kooplearn`` models can

1. Predict the evolution of states *and* observables.
2. Estimate the eigenvalues and eigenfucntions of the learned evolution operators.
3. Compute the `dynamic mode decomposition <https://en.wikipedia.org/wiki/Dynamic_mode_decomposition>`_ of states *and* observables.
4. Learn neural-network representations :math:`x_t \mapsto \varphi(x_t)` for evolution operators.

Why Choosing Kooplearn?
^^^^^^^^^^^^^^^^^^^^^^^

1. It is easy to use, and strictly adheres to the `scikit-learn API <https://scikit-learn.org/stable/api/index.html>`_.
2. :ref:`Kernel estimators <api_kernel>` are state-of-the-art: 
   
   * ``kooplearn`` implements the *Reduced Rank Regressor* from :cite:`Kostic2022` which is provably better :cite:`Kostic2023SpectralRates` than the classical kernel DMD :cite:`Williams2015_KDMD` in estimating eigenvalues and eigenfunctions. 
   * It also implements Nyström estimators :cite:`Meanti2023` and randomized estimators :cite:`turri2023randomized` for :doc:`blazingly fast <examples/ala2_nys_tutorial.ipynb>` kernel learning.

3. Includes representation learning losses to train neural-network Koopman embeddings.



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

To be able to use the representation-learning losses in ``kooplearn.nn``, run

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

   from kooplearn.datasets import make_duffing
   from kooplearn.kernel import KernelRidge
   import numpy as np

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

      .. card:: :material-regular:`library_books;2em` Guides
         :class-card: sd-text-black sd-bg-light sd-text-center sd-align-middle
         :link: guides/index.html

   .. grid-item::
      :columns: 6

      .. card:: :material-regular:`science;2em` Examples
         :class-card: sd-text-black sd-bg-light sd-text-center sd-align-middle
         :link: examples/index.html


   .. grid-item::
      :columns: 6

      .. card:: :material-regular:`menu_book;2em` API reference
         :class-card: sd-text-black sd-bg-light sd-text-center sd-align-middle
         :link: api_reference/index.html

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

   Estimating Eigenfunctions <examples/prinz_potential.ipynb>
   Fast Kernel Models <examples/kernel_methods.ipynb>
   Molecular Dynamics (Kernels) <examples/ala2_nys_tutorial.ipynb>
   Ordered MNIST (torch) <examples/ordered_mnist_torch.ipynb>
   Switching System <examples/switching_system.ipynb>


.. toctree::
   :maxdepth: 4
   :caption: API
   :hidden:

   api/linear_model
   api/kernel
   api/datasets
   
.. bibliography::
   :filter: docname in docnames
   :style: unsrt