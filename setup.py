
"""
stateinterpreter
Interpretation of metastable states from MD simulations
"""
import setuptools
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import glob
import os


c_ext_modules=[
    Extension(name="kerneldmd._kernel_helpers",
            sources=["kerneldmd/_kernel_helpers.pyx"],
            libraries=["m"],
            include_dirs=[numpy.get_include()],
            extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
            extra_link_args=['-fopenmp']
    )
]
eiscor_sources = glob.glob("eiscor/src/*/*.f90")
eiscor_modules = [os.path.dirname(f).split("/")[-1] for f in eiscor_sources]
eiscor_routines = [os.path.splitext(os.path.basename(f))[0] for f in eiscor_sources]
for m in eiscor_modules:
    dir_name = "kerneldmd/eiscor_wrapper/" + m + "/"
    if os.path.exists(dir_name):
        os.makedirs(dir_name)

f_ext_modules = []
for idx in range(len(eiscor_sources)):
    s, m, r = eiscor_sources[idx], eiscor_modules[idx], eiscor_routines[idx]
    if r == 'z_poly_roots':
        f_ext_modules.append(
            Extension(name="kerneldmd.eiscor_wrapper." + m + "." + r,
                sources= eiscor_sources,
                include_dirs=['eiscor/include', ],
                extra_compile_args=["-O3"],  
                extra_f90_compile_args = ["-O3", "-std=f95", "-cpp", "-fPIC", "-c"] 
            )
        )
setup(
    name='kerneldmd',
    ext_modules = f_ext_modules,
    zip_safe = False,
)
