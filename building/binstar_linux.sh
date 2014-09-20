#!/bin/bash
cd ..

sudo Xvfb :99 -nolisten tcp -fbdir /var/run

## declare Python and Numpy Versions
declare -a PY_VERSIONS=( "27" "33" "34" )
declare -a NPY_VERSIONS=( "18" "19" )

## Loop across Python and Numpy
for PY in "${PY_VERSIONS[@]}"
do
    export CONDA_PY=$PY
    for NPY in "${NPY_VERSIONS[@]}"
    do
        export CONDA_NPY=$NPY
        binstar remove bashtage/arch/1.0/linux-64/arch-1.0-np${NPY}py${PY}_0.tar.bz2 -f
        conda build ./building/binstar
    done
done
