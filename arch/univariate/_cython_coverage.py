#!/usr/bin/env python3

# Usage: python _cython_coverage.py build_ext --inplace


from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
import numpy as np

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
        "recursions",
        ["recursions.pyx"],
        define_macros=MACROS,
        include_dirs=[np_inc],
    )
]

setup(
    ext_modules=cythonize(ext_modules, force=True, compiler_directives=DIRECTIVES)
)
