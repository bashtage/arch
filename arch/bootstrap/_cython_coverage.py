#!/usr/bin/env python3

# Usage: python _cython_coverage.py build_ext --inplace


from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup

np_inc = np.get_include()
DIRECTIVES = {
    "language_level": "3",
    "cpow": True,
    "linetrace": True,
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "binding": True,
}
MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("CYTHON_TRACE", "1")]

ext_modules = [
    Extension(
        "_samplers",
        ["_samplers.pyx"],
        define_macros=MACROS,
        include_dirs=[np_inc],
    )
]

setup(ext_modules=cythonize(ext_modules, force=True, compiler_directives=DIRECTIVES))
