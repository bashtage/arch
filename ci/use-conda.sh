#!/usr/bin/env bash

PKGS="python=${PYTHON}"
PKGS="${PKGS} numpy"; if [[ -n ${NUMPY} ]]; then PKGS="${PKGS}=${NUMPY}"; fi;
PKGS="${PKGS} scipy"; if [[ -n ${SCIPY} ]]; then PKGS="${PKGS}=${SCIPY}"; fi;
PKGS="${PKGS} patsy"; if [[ -n ${PATSY} ]]; then PKGS="${PKGS}=${PATSY}"; fi;
PKGS="${PKGS} pandas"; if [[ -n ${PANDAS} ]]; then PKGS="${PKGS}=${PANDAS}"; fi;
PKGS="${PKGS} Cython"; if [[ -n ${CYTHON} ]]; then PKGS="${PKGS}=${CYTHON}"; fi;
if [[ -n ${MATPLOTLIB} ]]; then
  PKGS="${PKGS} matplotlib=${MATPLOTLIB} seaborn";
fi;
PKGS="${PKGS} statsmodels"; if [[ -n ${STATSMODELS} ]]; then PKGS="${PKGS}=${STATSMODELS}"; fi;
if [[ ${USE_NUMBA} = true ]]; then
  PKGS="${PKGS} numba";
  if [[ -n ${NUMBA} ]]; then
    PKGS="${PKGS}=${NUMBA}";
  fi;
fi;

wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh -nv
chmod +x miniconda3.sh
./miniconda3.sh -b
export PATH=/home/travis/miniconda3/bin:$PATH
conda config --set always_yes true
conda update --all --quiet
echo conda create --yes --quiet -n arch-test ${PKGS}
conda create --yes --quiet -n arch-test ${PKGS}
source activate arch-test
python -m pip install pip --upgrade
