from setuptools import Command, Extension, find_packages, setup
from setuptools.dist import Distribution

from collections import defaultdict
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
import fnmatch
import os
import sys

import pkg_resources

import versioneer

CYTHON_COVERAGE = os.environ.get("ARCH_CYTHON_COVERAGE", "0") in ("true", "1", "True")
if CYTHON_COVERAGE:
    print(
        "Building with coverage for cython modules, ARCH_CYTHON_COVERAGE="
        + os.environ["ARCH_CYTHON_COVERAGE"]
    )

try:
    from Cython.Build import cythonize
    from Cython.Distutils.build_ext import build_ext as _build_ext

    CYTHON_INSTALLED = True
except ImportError:
    CYTHON_INSTALLED = False
    if CYTHON_COVERAGE:
        raise ImportError(
            "cython is required for cython coverage. Unset " "ARCH_CYTHON_COVERAGE"
        )

    from setuptools.command.build_ext import build_ext as _build_ext

FAILED_COMPILER_WARNING = """
******************************************************************************
*                               WARNING                                      *
******************************************************************************

Unable to build binary modules for arch.  While these are not required to run
any code in the package, it is strongly recommended to either compile the
extension modules or to use numba.

******************************************************************************
*                               WARNING                                      *
******************************************************************************
"""

cmdclass = versioneer.get_cmdclass()


# prevent setup.py from crashing by calling import numpy before
# numpy is installed
class build_ext(_build_ext):
    def build_extensions(self) -> None:
        numpy_incl = pkg_resources.resource_filename("numpy", "core/include")

        for ext in self.extensions:
            if hasattr(ext, "include_dirs") and numpy_incl not in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


SETUP_REQUIREMENTS = {"numpy": "1.14", "cython": "0.29.14"}
INSTALL_REQUIREMENTS = SETUP_REQUIREMENTS.copy()
INSTALL_REQUIREMENTS.update(
    {
        "scipy": "1.2.3",
        "pandas": "0.23",
        "statsmodels": "0.10",
        "property_cached": "1.6.4",
    }
)

cmdclass["build_ext"] = build_ext


class BinaryDistribution(Distribution):
    def is_pure(self) -> bool:
        return False


class CleanCommand(Command):
    def run(self) -> None:
        raise NotImplementedError("Use git clean -xfd instead")

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass


cmdclass["clean"] = CleanCommand

with open("README.md", "r") as readme:
    description = readme.read()

package_data = defaultdict(list)
filetypes = ["*.csv", "*.csv.gz"]
for root, _, filenames in os.walk(os.path.join(os.getcwd(), "arch")):  # noqa: E501
    matches = []
    for filetype in filetypes:
        for filename in fnmatch.filter(filenames, filetype):
            matches.append(filename)
    if matches:
        package_data[".".join(os.path.relpath(root).split(os.path.sep))] = filetypes
package_data["arch"].append("py.typed")


def run_setup(binary: bool = True) -> None:
    if not binary:
        extensions = []
    else:
        directives = {"linetrace": CYTHON_COVERAGE}
        macros = [("NPY_NO_DEPRECATED_API", "1"), ("NPY_1_7_API_VERSION", "1")]
        if CYTHON_COVERAGE:
            macros.append(("CYTHON_TRACE", "1"))

        ext_modules = []
        ext_modules.append(
            Extension(
                "arch.univariate.recursions",
                ["./arch/univariate/recursions.pyx"],
                define_macros=macros,
            )
        )
        ext_modules.append(
            Extension(
                "arch.bootstrap._samplers",
                ["./arch/bootstrap/_samplers.pyx"],
                define_macros=macros,
            )
        )
        extensions = cythonize(
            ext_modules, force=CYTHON_COVERAGE, compiler_directives=directives
        )

    setup(
        name="arch",
        license="NCSA",
        version=versioneer.get_version(),
        description="ARCH for Python",
        long_description=description,
        long_description_content_type="text/markdown",
        author="Kevin Sheppard",
        author_email="kevin.sheppard@economics.ox.ac.uk",
        url="http://github.com/bashtage/arch",
        packages=find_packages(),
        ext_modules=extensions,
        package_dir={"arch": "./arch"},
        cmdclass=cmdclass,
        keywords=[
            "arch",
            "ARCH",
            "variance",
            "econometrics",
            "volatility",
            "finance",
            "GARCH",
            "bootstrap",
            "random walk",
            "unit root",
            "Dickey Fuller",
            "time series",
            "confidence intervals",
            "multiple comparisons",
            "Reality Check",
            "SPA",
            "StepM",
        ],
        zip_safe=False,
        include_package_data=False,
        package_data=package_data,
        distclass=BinaryDistribution,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: End Users/Desktop",
            "Intended Audience :: Financial and Insurance Industry",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "License :: OSI Approved",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Programming Language :: Python",
            "Programming Language :: Cython",
            "Topic :: Scientific/Engineering",
        ],
        install_requires=[
            key + ">=" + INSTALL_REQUIREMENTS[key] for key in INSTALL_REQUIREMENTS
        ],
        setup_requires=[
            key + ">=" + SETUP_REQUIREMENTS[key] for key in SETUP_REQUIREMENTS
        ],
        python_requires=">=3.6",
    )


try:
    build_binary = "--no-binary" not in sys.argv and CYTHON_INSTALLED
    if "--no-binary" in sys.argv:
        sys.argv.remove("--no-binary")

    run_setup(binary=build_binary)
except (
    CCompilerError,
    DistutilsExecError,
    DistutilsPlatformError,
    IOError,
    ValueError,
):
    run_setup(binary=False)
    import warnings

    warnings.warn(FAILED_COMPILER_WARNING, UserWarning)
