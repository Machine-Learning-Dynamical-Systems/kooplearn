Getting Started with kooplearn
==============================

Welcome to the "Getting Started" page for the Python package kooplearn! This package implement various kernel-based algorithms to learn the Koopman operator, which is a mathematical tool used to study dynamical systems.

Installation
------------

Before you can use kooplearn, you'll need to install it. You can do so using ``pip``, the Python package manager::
    
    pip install kooplearn

Kooplearn requires a minimal set of dependencies to work. Tu unlock the GPU support one can install it with the following command::

    pip install kooplearn[gpu]

Kooplearn also has a set of JAX-compatible estimators for the primal problem that can be installed with the following command::

    pip install kooplearn[jax]

..as well as a set of torch-compatible estimators for the primal problem that can be installed with the following command::

    pip install kooplearn[torch]

Basic Usage
-----------

Once you've installed kooplearn, you can import it in your Python code and start using it. Here's an example of how to use the PrincipalComponent class to learn the Koopman operator:

.. code-block:: python
    :linenos:

    import numpy as np
    from kooplearn.estimators import PrincipalComponent
    from kooplearn.kernels import RBF

    # Generate some sample data
    t = np.linspace(0, 10, 1000)
    x = np.sin(t)
    y = np.cos(t)

    # Concatenate the data into a matrix
    data = np.vstack([x, y]).T

    kernel = RBF(length_scale=1.0)

    # Create a PrincipalComponent object with a Gaussian kernel
    koop = PrincipalComponent(kernel=kernel, rank=1)

    # Fit the Koopman operator to the data
    koop.fit(data[:-1], data[1:]])

    # Evaluate the Koopman operator at a new point
    z = np.array([[0.5, 0.5]])
    Kz = koop.predict(z)

in this example, we generate some sample data consisting of two sine waves. We then concatenate this data into a matrix, `data`, which has shape `(1000, 2)`.

Next, we create a RBF kernel object with `length_scale` parameter of 1.0 and a `PrincipalComponent` object. We fit the Koopman operator to the data using the fit() method, which returns a fitted Koopman operator.

Finally, we evaluate the Koopman operator at a new point, `[0.5, 0.5]`, using the `predict()` method.

Documentation
-------------

The documentation includes more examples, a detailed explanation of the various classes and functions in the package, and information on how to define custom the kernels used by kooplearn.

Conclusion
----------

In this "Getting Started" guide, we've shown you how to install kooplearn and use the `PrincipalComponent` class to learn the Koopman operator. We hope this guide has been helpful, and we encourage you to explore the package further and experiment with different kernel functions to see how they affect the results.
