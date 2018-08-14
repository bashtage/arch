from __future__ import print_function

import glob
import os
import re
import subprocess
import sys
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from distutils.version import StrictVersion

import pkg_resources
import versioneer
from Cython.Build import cythonize
from setuptools import Command, Extension, find_packages, setup
from setuptools.dist import Distribution

CYTHON_COVERAGE = os.environ.get('ARCH_CYTHON_COVERAGE', '0') in ('true', '1', 'True')
if CYTHON_COVERAGE:
    print('Building with coverage for cython modules, ARCH_CYTHON_COVERAGE=' +
          os.environ['ARCH_CYTHON_COVERAGE'])

try:
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
REQUIREMENTS = {'Cython': '0.24',
                'matplotlib': '1.5',
                'scipy': '0.19',
                'pandas': '0.20',
                'statsmodels': '0.8'}

ALL_REQUIREMENTS = SETUP_REQUIREMENTS.copy()
ALL_REQUIREMENTS.update(REQUIREMENTS)

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


def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)

cwd = os.getcwd()

# Convert markdown to rst for submission
long_description = ''
try:
    cmd = 'pandoc --from=markdown --to=rst --output=README.rst README.md'
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    long_description = open(os.path.join(cwd, "README.rst")).read()
except IOError as e:
    import warnings
    
    warnings.warn('Unable to convert README.md.  Most likely because pandoc '
                  'is not installed')

def run_setup(binary=True):
    if not binary:
        del REQUIREMENTS['Cython']
        extensions = []
    else:
        directives = {'linetrace': CYTHON_COVERAGE}
        macros = []
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
          long_description=long_description,
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
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'License :: OSI Approved',
              'Operating System :: MacOS :: MacOS X',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Programming Language :: Python',
              'Programming Language :: Cython',
              'Topic :: Scientific/Engineering',
          ],
          install_requires=[key + '>=' + REQUIREMENTS[key]
                            for key in REQUIREMENTS],
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
