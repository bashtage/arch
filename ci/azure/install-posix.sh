#!/usr/bin/env bash

python -m pip install cython pytest pytest-xdist coverage pytest-cov ipython jupyter notebook nbconvert "property_cached>=1.6.3" black==20.8b1 isort flake8

CMD="python -m pip install numpy"
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
