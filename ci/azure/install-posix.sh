#!/usr/bin/env bash

if [[ ${USE_CONDA} == "true" ]]; then
  conda config --set always_yes true
  conda update --all --quiet
  conda create -n arch-test python=${PYTHON_VERSION} -y
  conda activate arch-test
  conda init
  echo ${PATH}
  source activate arch-test
  echo ${PATH}
  which python
  CMD="conda install numpy"
else
  CMD="python -m pip install numpy"
fi

python -m pip install --upgrade pip "setuptools>=61" wheel
python -m pip install cython "pytest>=9,<10" pytest-xdist "coverage[toml]" pytest-cov ipython jupyter notebook nbconvert "property_cached>=1.6.3" nbconvert setuptools_scm colorama "meson-python>=0.18.0" meson ninja


# Versions are exact so that the job tests what its matrix entry says, and all of
# the packages are installed in a single command so that the installer picks
# versions of matplotlib, seaborn, ... that work with them
if [[ -n ${NUMPY} ]]; then CMD="$CMD==${NUMPY}"; fi;
CMD="$CMD scipy"
if [[ -n ${SCIPY} ]]; then CMD="$CMD==${SCIPY}"; fi;
CMD="$CMD pandas"
if [[ -n ${PANDAS} ]]; then CMD="$CMD==${PANDAS}"; fi;
CMD="$CMD statsmodels"
if [[ -n ${STATSMODELS} ]]; then CMD="$CMD==${STATSMODELS}"; fi
# NO_MATPLOTLIB leaves out matplotlib and seaborn, which needs it, as the tests of the
# arch-wheels builds do
if [[ ${NO_MATPLOTLIB,,} != true ]]; then
  CMD="$CMD matplotlib"
  if [[ -n ${MATPLOTLIB} ]]; then CMD="$CMD==${MATPLOTLIB}"; fi
  CMD="$CMD seaborn"
fi
if [[ ${USE_NUMBA} = true ]]; then
  CMD="${CMD} numba";
  if [[ -n ${NUMBA} ]]; then
    CMD="${CMD}==${NUMBA}"
  fi;
fi;
CMD="$CMD $EXTRA"
echo $CMD
eval $CMD || exit 1

if [[ ${NO_MATPLOTLIB,,} == true ]]; then
  # Fail if another package installed matplotlib, so that the job really tests without it
  if python -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('matplotlib') is None)"; then
    echo "##vso[task.logissue type=error]matplotlib is installed although NO_MATPLOTLIB is true"
    exit 1
  fi
fi

if [ "${NIGHTLY}" = true ]; then
  # Install over the released versions, rather than uninstalling them first, so that
  # a failure of the (slow, sometimes unreachable) nightly index leaves a working
  # environment. The failure is shown as a warning.
  python -m pip install --pre --upgrade cython meson-python
  python -m pip install --pre --upgrade --no-deps --only-binary=:all: --retries 10 --timeout 60 -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple --extra-index-url https://pypi.org/simple numpy pandas scipy matplotlib statsmodels || echo "##vso[task.logissue type=warning]Installing the nightly builds failed"
fi
