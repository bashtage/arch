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
python -m pip install cython "pytest>=7,<7.1" pytest-xdist coverage pytest-cov ipython jupyter notebook nbconvert "property_cached>=1.6.3" black isort flake8 nbconvert setuptools_scm

if [[ -n ${NUMPY} ]]; then CMD="$CMD~=${NUMPY}"; fi;
CMD="$CMD scipy"
if [[ -n ${SCIPY} ]]; then CMD="$CMD~=${SCIPY}"; fi;
CMD="$CMD pandas"
if [[ -n ${PANDAS} ]]; then CMD="$CMD~=${PANDAS}"; fi;
CMD="$CMD statsmodels"
if [[ -n ${STATSMODELS} ]]; then CMD="$CMD~=${STATSMODELS}"; fi
if [[ -n ${MATPLOTLIB} ]]; then CMD="$CMD matplotlib~=${MATPLOTLIB} seaborn"; fi
if [[ ${USE_NUMBA} = true ]]; then
  CMD="${CMD} numba";
  if [[ -n ${NUMBA} ]]; then
    CMD="${CMD}~=${NUMBA}"
  fi;
fi;
CMD="$CMD $EXTRA"
echo $CMD
eval $CMD
