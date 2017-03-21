#!/bin/bash

conda install anaconda-client conda-build --yes
conda config --set anaconda_upload yes

cd ..

export VERSION=4.1
## detect OS
if [ "$(uname)" == "Darwin" ]; then
    export OS=osx-64
else
    export OS=linux-64
fi

# No longer needed
## sudo Xvfb :99 -nolisten tcp -fbdir /var/run &

## declare Python and Numpy Versions
declare -a PY_VERSIONS=( "27" "35" "36" )
declare -a NPY_VERSIONS=( "111" "112" )

## Loop across Python and Numpy
for PY in "${PY_VERSIONS[@]}"
do
    export CONDA_PY=${PY}
    for NPY in "${NPY_VERSIONS[@]}"
    do
        export CONDA_NPY=${NPY}
        anaconda remove bashtage/arch/${VERSION}/${OS}/arch-${VERSION}-np${NPY}py${PY}_0.tar.bz2 -f
        conda build --python ${PY} --numpy ${NPY} ./building/binstar
    done
done