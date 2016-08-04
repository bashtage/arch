#!/bin/bash
cd ..

export VERSION=3.2
## detect OS
if [ "$(uname)" == "Darwin" ]; then
    export OS=osx-64
else
    export OS=linux-64
fi

sudo Xvfb :99 -nolisten tcp -fbdir /var/run &

## declare Python and Numpy Versions
declare -a PY_VERSIONS=( "27" "35" )
declare -a NPY_VERSIONS=( "110" "111" )

## Loop across Python and Numpy
for PY in "${PY_VERSIONS[@]}"
do
    export CONDA_PY=$PY
    for NPY in "${NPY_VERSIONS[@]}"
    do
        export CONDA_NPY=$NPY
        binstar remove bashtage/arch/${VERSION}/${OS}/arch-${VERSION}-np${NPY}py${PY}_0.tar.bz2 -f
        conda build ./building/binstar
    done
done