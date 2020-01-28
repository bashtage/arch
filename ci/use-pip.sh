#!/usr/bin/env bash

PKGS="numpy"; if [[ -n ${NUMPY} ]]; then PKGS="${PKGS}==${NUMPY}"; fi;
PKGS="${PKGS} scipy"; if [[ -n ${SCIPY} ]]; then PKGS="${PKGS}==${SCIPY}"; fi;
PKGS="${PKGS} patsy"; if [[ -n ${PATSY} ]]; then PKGS="${PKGS}==${PATSY}"; fi;
PKGS="${PKGS} pandas"; if [[ -n ${PANDAS} ]]; then PKGS="${PKGS}==${PANDAS}"; fi;
PKGS="${PKGS} Cython"; if [[ -n ${CYTHON} ]]; then PKGS="${PKGS}==${CYTHON}"; fi;
if [[  -n ${MATPLOTLIB} ]]; then
  PKGS="${PKGS} matplotlib==${MATPLOTLIB} seaborn";
fi;
PKGS="${PKGS} statsmodels"; if [[ -n ${STATSMODELS} ]]; then PKGS="${PKGS}==${STATSMODELS}"; fi;
if [[ ${USE_NUMBA} = true ]]; then
  PKGS="${PKGS} numba";
  if [[ -n ${NUMBA} ]]; then
    PKGS="${PKGS}==${NUMBA}";
  fi;
fi;

python -m pip install pip --upgrade

echo python -m pip install ${PKGS}
python -m pip install ${PKGS}
