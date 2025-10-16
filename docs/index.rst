.. kooplearn documentation master file, created by
   sphinx-quickstart on Wed Oct 15 05:56:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

kooplearn documentation
=======================

.. div:: sd-text-left sd-font-italic

   A sklearn-compatible library for Koopman and Transfer operator learning.

----

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
   :caption: Examples
   :hidden:

   Getting Started <examples/linear_system.md>
   Large Scale Kernel Methods <examples/kernel_methods.md>
   Molecular Dynamics (Kernels) <examples/ala2_nys_tutorial.md>

.. autosummary::
   :caption: API reference
   :toctree: _autosummary
   :template: custom-module-template.rst

   kooplearn.kernel
   kooplearn.linear_model
   kooplearn.datasets
