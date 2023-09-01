.. kooplearn documentation master file, created by
   sphinx-quickstart on Sun Aug 20 13:35:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************************
Kooplearn
******************************


.. div:: sd-text-left sd-font-italic

   Koopman and Transfer operator learning in Python

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

            Kooplearn offers a diverse range of state of the art algorithms tailored for learning Koopman operators and transfer operators. These algorithms cover the most important kernel-based and neural network-based methods.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Forecasting
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Kooplearn implements forecasting capabilities out of the box. Harness the learned operators to predict the future behavior of any observable of the dynamical system.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Eigenvalue Decomposition
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Unlock deeper insights into your dynamical systems using spectral analysis. Kooplearn implements eigenvalue decomposition, and a dedicated visualization software, helping you uncover critical system modes and understand the fundamental dynamics driving your data.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Modularity
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Kooplearn is designed with extensibility in mind. In :mod:`kooplearn.abc <kooplearn.abc>` we expose simple abstract base classes which allow you to quickly build kooplearn-compatible components.

----

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install kooplearn
   # Or to install the latest version of kooplearn from git:
   pip install --upgrade git+https://github.com/CSML-IIT-UCL/kooplearn.git


Basic usage
^^^^^^^^^^^^
TODO: add basic usage example 

----

Dashboard
^^^^^^^^^^^^
TODO: add dashboard example with .gif animation

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
   
   getting_started
   examples/index
   datasets
   guides/index
   api_reference/index

