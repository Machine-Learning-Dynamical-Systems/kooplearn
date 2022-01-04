
"""
stateinterpreter
Interpretation of metastable states from MD simulations
"""
import sys
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy


ext_modules=[
    Extension("_kernel_helpers",
            ["_kernels.pyx"],
            libraries=["m"],
            include_dirs=[numpy.get_include()],
            extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
            extra_link_args=['-fopenmp']
    ) 
]

setup(
    ext_modules = cythonize(ext_modules),
    zip_safe = False,
)