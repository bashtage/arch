#!/usr/bin/env bash

conda remove statsmodels --yes
export GITDIR=$PWD
cd ~
git clone --branch=master --depth=10 https://github.com/statsmodels/statsmodels.git
cd statsmodels
python setup.py install
cd $GITDIR
pip install seaborn
