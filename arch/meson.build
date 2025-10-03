incdir_numpy = run_command(
    py,
    [
        '-c',
        '''
import os
cwd = os.getcwd()

# Protect from import errors due to module names
os.chdir(os.path.join('..', 'examples'))
import numpy as np
os.chdir(cwd)

try:
    # Check if include directory is inside the dir
    # e.g. a venv created inside the dir
    # If so, convert it to a relative path
    incdir = os.path.relpath(np.get_include())
except Exception as exc:
    incdir = np.get_include()
print(incdir)
     ''',
    ],
    check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy, is_system: true)

# Copy the main __init__.py to the build dir.
# Some submodules (linalg, special, optimize) add pxd files to this.
# Needed to trick Cython, it won't do a relative import outside a package
_cython_tree = [fs.copyfile('__init__.py')]
cython_args = [
    '-X cpow=True',
    '-X boundscheck=False',
    '-X wraparound=False',
    '-X cdivision=True',
    '-X binding=True'
]
cython_c_args =  ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']
if get_option('cython-coverage')
  message('Building with Cython coverage support')
  cython_args += ['-X linetrace=True']
  cython_c_args += ['-DCYTHON_TRACE=1']
endif

arch_dir = py.get_install_dir() / 'arch'
# Generate version.py for sdist
meson.add_dist_script(
   ['_build/git_version.py', '--meson-dist', '--write',
     'arch/_version.py']
)
if not fs.exists('_version.py')
  generate_version = custom_target(
    'generate-version',
    install: true,
    build_always_stale: true,
    build_by_default: true,
    output: '_version.py',
    input: '_build/git_version.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    install_dir: arch_dir,
    install_tag: 'python-runtime'
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['_version.py'],
    subdir : 'arch'
  )
endif

subdir('bootstrap')
subdir('univariate')

subdirs_list = [
    '__future__',
    '_build',
    'bootstrap',
    'compat',
    'covariance',
    'data',
    'tests',
    'unitroot',
    'univariate',
    'utility',
    'vendor',
]

foreach subdir : subdirs_list
    install_subdir(subdir, install_dir: py.get_install_dir() / 'arch')
endforeach

top_level_py_list = [
    '__init__.py',
    'conftest.py',
    'py.typed',
    '_typing.py'
]
py.install_sources(top_level_py_list, subdir: 'arch')
