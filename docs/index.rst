.. kooplearn documentation master file, created by
   sphinx-quickstart on Sun Aug 20 13:35:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************************
Kooplearn
******************************


.. div:: sd-text-left sd-font-italic

   A Python library for Koopman and Transfer operator learning

----

Installation
^^^^^^^^^^^^

To install the core version of ``kooplearn``, without optional the dependencies `Torch <https://pytorch.org/>`_ and `Lightning <https://lightning.ai/>`_, run

.. code-block:: bash

   pip install kooplearn

To install the full version of ``kooplearn``, including Neural-Network models, run

.. code-block:: bash

   pip install "kooplearn[full]"

To install the development version of ``kooplearn``, run

.. code-block:: bash

   pip install --upgrade git+https://github.com/Machine-Learning-Dynamical-Systems/kooplearn.git

----

Features
^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Algorithms
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Kooplearn offers a diverse range of state of the art algorithms tailored for learning :ref:`Koopman and Transfer operators <primer>` of deterministic and stochastic dynamical systems, respectively. Check out kooplearn's :ref:`model zoo <model_zoo>` for a complete list of available algorithms.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Modularity
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Kooplearn is designed with extensibility in mind. In :mod:`kooplearn.abc <kooplearn.abc>` we expose simple abstract base classes which allow you to quickly build kooplearn-compatible components and models.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Spectral Decomposition
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Unlock deeper insights into your dynamical systems using spectral analysis. Every model in kooplearn implements :code:`eig` and :code:`modes` methods returning the spectral and mode decompositions of the learned operator. This can be used for a number of downstream tasks, such as control, system identification, and more.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Deep Learning models
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Kooplearn implements many neural-network models to learn the Koopman/transfer operators. Kooplearn's Deep-Learning models are based upon `Pytorch Lightning <https://lightning.ai/>`_ for fast and easy training on CPU, GPU, and multi-GPU systems. 

----


Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Getting started
         :class-card: sd-text-black sd-bg-light
         :link: getting_started.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Guides
         :class-card: sd-text-black sd-bg-light
         :link: guides/index.html

   ..
      .. grid-item::
         :columns: 6 6 6 4

         .. card:: :material-regular:`science;2em` Examples
            :class-card: sd-text-black sd-bg-light
            :link: examples/index.html


   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`menu_book;2em` API reference
         :class-card: sd-text-black sd-bg-light
         :link: api_reference/index.html

----

.. toctree::
   :maxdepth: 1
   
   primer
   getting_started
   examples/index
   model_zoo
   datasets
   guides/index
   api_reference/index

