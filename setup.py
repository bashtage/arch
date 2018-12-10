from __future__ import print_function

import os
import sys
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

import pkg_resources
from setuptools import Command, Extension, find_packages, setup
from setuptools.dist import Distribution

import versioneer

CYTHON_COVERAGE = os.environ.get('ARCH_CYTHON_COVERAGE', '0') in ('true', '1', 'True')
if CYTHON_COVERAGE:
    print('Building with coverage for cython modules, ARCH_CYTHON_COVERAGE=' +
          os.environ['ARCH_CYTHON_COVERAGE'])

try:
    from Cython.Build import cythonize
    from Cython.Distutils.build_ext import build_ext as _build_ext

    CYTHON_INSTALLED = True
except ImportError:
    CYTHON_INSTALLED = False
    if CYTHON_COVERAGE:
        raise ImportError('cython is required for cython coverage. Unset '
                          'ARCH_CYTHON_COVERAGE')

    class _build_ext(object):
        pass

FAILED_COMPILER_ERROR = """
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
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')

        for ext in self.extensions:
            if (hasattr(ext, 'include_dirs') and
                    numpy_incl not in ext.include_dirs):
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)


SETUP_REQUIREMENTS = {'numpy': '1.12'}
INSTALL_REQUIREMENTS = SETUP_REQUIREMENTS.copy()
INSTALL_REQUIREMENTS.update({'scipy': '0.19',
                             'pandas': '0.20',
                             'statsmodels': '0.8',
                             'cached_property': '1.5.1'})

cmdclass['build_ext'] = build_ext


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


class CleanCommand(Command):
    user_options = []

    def run(self):
        raise NotImplementedError('Use git clean -xfd instead')

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


cmdclass['clean'] = CleanCommand

try:
    markdown = os.stat('README.md').st_mtime
    if os.path.exists('README.rst'):
        rst = os.stat('README.rst').st_mtime
    else:
        rst = markdown - 1

    if rst >= markdown:
        with open('README.rst', 'r') as rst:
            description = rst.read()
    else:
        import pypandoc

        osx_line_ending = '\r'
        windows_line_ending = '\r\n'
        linux_line_ending = '\n'

        description = pypandoc.convert_file('README.md', 'rst+smart')
        description = description.replace(windows_line_ending, linux_line_ending)
        description = description.replace(osx_line_ending, linux_line_ending)
        with open('README.rst', 'w') as rst:
            rst.write(description)
except (ImportError, OSError):
    import warnings

    warnings.warn("Unable to convert README.md to README.rst", UserWarning)
    description = open('README.md').read()


def run_setup(binary=True):
    if not binary:
        extensions = []
    else:
        directives = {'linetrace': CYTHON_COVERAGE}
        macros = [('NPY_NO_DEPRECATED_API', '1'),
                  ('NPY_1_7_API_VERSION', '1')]
        if CYTHON_COVERAGE:
            macros.append(('CYTHON_TRACE', '1'))

        ext_modules = []
        ext_modules.append(Extension("arch.univariate.recursions",
                                     ["./arch/univariate/recursions.pyx"],
                                     define_macros=macros))
        ext_modules.append(Extension("arch.bootstrap._samplers",
                                     ["./arch/bootstrap/_samplers.pyx"],
                                     define_macros=macros))
        extensions = cythonize(ext_modules,
                               force=CYTHON_COVERAGE,
                               compiler_directives=directives)

    setup(name='arch',
          license='NCSA',
          version=versioneer.get_version(),
          description='ARCH for Python',
          long_description=description,
          author='Kevin Sheppard',
          author_email='kevin.sheppard@economics.ox.ac.uk',
          url='http://github.com/bashtage/arch',
          packages=find_packages(),
          ext_modules=extensions,
          package_dir={'arch': './arch'},
          cmdclass=cmdclass,
          keywords=['arch', 'ARCH', 'variance', 'econometrics', 'volatility',
                    'finance', 'GARCH', 'bootstrap', 'random walk', 'unit root',
                    'Dickey Fuller', 'time series', 'confidence intervals',
                    'multiple comparisons', 'Reality Check', 'SPA', 'StepM'],
          zip_safe=False,
          include_package_data=True,
          distclass=BinaryDistribution,
          classifiers=[
              'Development Status :: 5 - Production/Stable',
              'Intended Audience :: End Users/Desktop',
              'Intended Audience :: Financial and Insurance Industry',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'License :: OSI Approved',
              'Operating System :: MacOS :: MacOS X',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Programming Language :: Python',
              'Programming Language :: Cython',
              'Topic :: Scientific/Engineering',
          ],
          install_requires=[key + '>=' + INSTALL_REQUIREMENTS[key]
                            for key in INSTALL_REQUIREMENTS],
          setup_requires=[key + '>=' + SETUP_REQUIREMENTS[key]
                          for key in SETUP_REQUIREMENTS],
          )


try:
    build_binary = '--no-binary' not in sys.argv and CYTHON_INSTALLED
    if '--no-binary' in sys.argv:
        sys.argv.remove('--no-binary')

    run_setup(binary=build_binary)
except (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, ValueError):
    run_setup(binary=False)
    import warnings

    warnings.warn(FAILED_COMPILER_ERROR, UserWarning)
