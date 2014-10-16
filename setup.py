# setup.py
import os
import subprocess
import sys
import re
from distutils.version import StrictVersion

from setuptools import setup, Extension, find_packages
from setuptools.dist import Distribution
from Cython.Distutils import build_ext

REQUIREMENTS = {'Cython': '0.20',
                'matplotlib': '1.2',
                'numpy': '1.7',
                'scipy': '0.12',
                'pandas': '0.12',
                'statsmodels': '0.5',
                'patsy': '0.2'}

ext_modules = []
if not '--no-binary' in sys.argv:
    ext_modules.append(Extension("arch.univariate.recursions",
                                 ["./arch/univariate/recursions.pyx"]))
    ext_modules.append(Extension("arch.bootstrap._samplers",
                                 ["./arch/bootstrap/_samplers.pyx"]))
else:
    sys.argv.remove('--no-binary')


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


def strip_rc(version):
    return re.sub(r"rc\d+$", "", version)

# Polite checks for numpy, scipy and pandas.  These should not be upgraded,
# and if found and below the required version, refuse to install
missing_package = '{package} is installed, but the version installed' \
                  ', {existing_ver}, is less than the required version ' \
                  'of {required_version}. This package must be manually ' \
                  'updated.  If this isn\'t possible, consider installing in ' \
                  'an empty virtual environment.'

PACKAGE_CHECKS = ['numpy','scipy','pandas']
for key in PACKAGE_CHECKS:
    version = None
    satisfies_req = True
    existing_version = 'Too Old to Detect'
    if key == 'numpy':
        try:
            import numpy
            try:
                from numpy.version import short_version as version
            except ImportError:
                satisfies_req = False
        except ImportError:
            pass

    elif key == 'scipy':
        try:
            import scipy
            try:
                from scipy.version import short_version as version
            except ImportError:
                satisfies_req = False
        except ImportError:
            pass

    elif key == 'pandas':
        try:
            from pandas.version import short_version as version
        except ImportError:
            pass
        except:  # very old version
            satisfies_req = False
    else:
        raise NotImplementedError('Unknown package')

    if version:
        existing_version = StrictVersion(strip_rc(version))
        satisfies_req = existing_version >= REQUIREMENTS[key]
    if not satisfies_req:
        message = missing_package.format(package=key,
                                         existing_ver=existing_version,
                                         required_version=REQUIREMENTS[key])
        raise EnvironmentError(message)

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

    warnings.warn('Unable to convert README.md.  Most likely because pandoc is '
                  'not installed')

# Convert examples notebook to rst for docs
try:
    from IPython.nbconvert.utils.exceptions import ConversionException
    from IPython.nbconvert.utils.pandoc import PandocMissing
except ImportError as e:
    ConversionException = PandocMissing = ValueError

try:
    from IPython.nbformat import current as nbformat
    from IPython.nbconvert import RSTExporter
    import glob
    notebooks = glob.glob(os.path.join(cwd, 'examples', '*.ipynb'))
    for notebook in notebooks:
        f = open(notebook, 'rt')
        example_nb = f.read()
        f.close()

        example_nb = nbformat.reads_json(example_nb)
        rst_export = RSTExporter()
        (body, resources) = rst_export.from_notebook_node(example_nb)
        rst_path = os.path.join(cwd, 'doc', 'source')
        path_parts = os.path.split(notebook)
        nb_filename = path_parts[-1]
        nb_filename = nb_filename.split('.')[0]
        source_dir = nb_filename.split('_')[0]
        rst_filename = os.path.join(cwd, 'doc', 'source',
                                    source_dir,  nb_filename + '.rst')
        f = open(rst_filename, 'wt')
        f.write(body)
        f.close()
        for key in resources['outputs'].keys():
            if key.endswith('.png'):
                resource_filename = os.path.join(cwd, 'doc', 'source',
                                                 source_dir, key)
                f = open(resource_filename, 'wb')
                f.write(resources['outputs'][key])
                f.close()
except:
    import warnings

    warnings.warn('Unable to convert examples.ipynb to examples.rst.  This only'
                  'affects documentation generation and not the operation of the'
                  ' module.')

# Read version information from plain VERSION file
version = None
try:
    version_file = open(os.path.join(cwd, 'VERSION'), 'rt')
    version = version_file.read().strip()
    version_file.close()
    version_py_file = open(os.path.join(cwd, 'arch', '_version.py'), mode='wt')
    version_py_file.write('__version__="' + version + '"')
    version_py_file.close()
except:
    raise EnvironmentError('Cannot locate VERSION')

setup(name='arch',
      license='NCAA',
      version=version,
      description='ARCH for Python',
      long_description=long_description,
      author='Kevin Sheppard',
      author_email='kevin.sheppard@economics.ox.ac.uk',
      url='http://github.com/bashtage/arch',
      packages=find_packages(),
      ext_modules=ext_modules,
      package_dir={'arch': './arch'},
      cmdclass={'build_ext': build_ext},
      include_dirs=[numpy.get_include()],
      keywords=['arch', 'ARCH', 'variance', 'econometrics', 'volatility',
                'finance', 'GARCH'],
      zip_safe=False,
      include_package_data=True,
      distclass=BinaryDistribution,
      install_requires=[key + '>=' + REQUIREMENTS[key] for key in REQUIREMENTS]
)
