# setup.py
import os
import subprocess
import sys

from setuptools import setup, Extension
from setuptools.dist import Distribution
from Cython.Distutils import build_ext
import numpy


ext_modules = []
if not '--no-binary' in sys.argv:
    ext_modules.append(Extension("arch.univariate.recursions",
                                 ["./arch/univariate/recursions.pyx"]))
else:
    sys.argv.remove('--no-binary')


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


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
    from IPython.nbformat import current as nbformat
    from IPython.nbconvert import RSTExporter

    f = open(os.path.join(cwd, 'examples', 'examples.ipynb'), 'rt')
    example_nb = f.read()
    f.close()

    example_nb = nbformat.reads_json(example_nb)
    rst_export = RSTExporter()
    (body, resources) = rst_export.from_notebook_node(example_nb)
    f = open(os.path.join(cwd, 'doc', 'source', 'examples.rst'), 'wt')
    f.write(body)
    f.close()
    for key in resources['outputs'].keys():
        if key.endswith('.png'):
            f = open(os.path.join(cwd, 'doc', 'source', key), 'wb')
            f.write(resources['outputs'][key])
            f.close()
except IOError as e:
    import warnings

    warnings.warn('Unable to convert examples.ipynb to examples.rst.  This only'
                  'affects documentation generation and not operation of the '
                  'module.')

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
      packages=['arch', 'arch.tests', 'arch.compat', 'arch.univariate'],
      ext_modules=ext_modules,
      package_dir={'arch': './arch'},
      cmdclass={'build_ext': build_ext},
      include_dirs=[numpy.get_include()],
      keywords=['arch', 'ARCH', 'variance', 'econometrics', 'volatility',
                'finance', 'GARCH'],
      zip_safe=False,
      include_package_data=True,
      distclass=BinaryDistribution,
)
