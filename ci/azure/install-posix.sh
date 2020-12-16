#!/usr/bin/env bash

if [[ ${USE_CONDA} == "true" ]]; then
  # TODO: Python version
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh -nv
  chmod +x miniconda3.sh
  ./miniconda3.sh -b -p $HOME/miniconda3
  conda config --set always_yes true
  conda update --all --quiet
  conda create -n arch-test python=${PYTHON_VERSION} -y
  conda activate arch-test
  which python
  CMD="conda install numpy"
  source ci/azure/update_path.sh
else
  CMD="python -m pip install numpy"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install cython pytest pytest-xdist coverage pytest-cov ipython jupyter notebook nbconvert "property_cached>=1.6.3" black==20.8b1 isort flake8 nbconvert==5.6.1

if [[ -n ${NUMPY} ]]; then CMD="$CMD==${NUMPY}"; fi;
CMD="$CMD scipy"
if [[ -n ${SCIPY} ]]; then CMD="$CMD==${SCIPY}"; fi;
CMD="$CMD pandas"
if [[ -n ${PANDAS} ]]; then CMD="$CMD==${PANDAS}"; fi;
CMD="$CMD statsmodels"
if [[ -n ${STATSMODELS} ]]; then CMD="$CMD==${STATSMODELS}"; fi
if [[ -n ${MATPLOTLIB} ]]; then CMD="$CMD matplotlib==${MATPLOTLIB} seaborn"; fi
if [[ ${USE_NUMBA} = true ]]; then
  CMD="${CMD} numba";
  if [[ -n ${NUMBA} ]]; then
    CMD="${CMD}==${NUMBA}"
  fi;
fi;
CMD="$CMD $EXTRA"
echo $CMD
eval $CMD
