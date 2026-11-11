#!/usr/bin/env bash

# The build tools, the test requirements and the packages that the example notebooks
# need are dependency groups in pyproject.toml, which pip 25.1 or later installs
DEP_GROUPS=(--group build --group test --group notebooks)

if [[ ${USE_CONDA} == "true" ]]; then
  # Miniforge (conda-forge only) is installed by the pipeline and the env's bin
  # directory is on PATH, so no activation is needed.
  mamba create -n arch-test -y python=${PYTHON_VERSION} || exit 1
  echo ${PATH}
  which python
  # Fail here rather than silently using a different Python
  if [[ "$(which python)" != */envs/arch-test/bin/python ]]; then
    echo "##vso[task.logissue type=error]python is not from the arch-test environment"
    exit 1
  fi
  CMD="mamba install -n arch-test -y -c conda-forge numpy"
else
  CMD="python -m pip install ${DEP_GROUPS[*]} numpy"
fi

python -m pip install --upgrade pip

# Versions are exact so that the job tests what its matrix entry says, and all of
# the packages (with pip, also the dependency groups) are installed in a single
# command so that the installer picks versions of matplotlib, seaborn, ... that work
# with them
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

if [[ ${USE_CONDA} == "true" ]]; then
  # After the conda packages, so that pip keeps the numpy from conda-forge
  echo python -m pip install "${DEP_GROUPS[@]}"
  python -m pip install "${DEP_GROUPS[@]}" || exit 1
fi

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
