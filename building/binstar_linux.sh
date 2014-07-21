#!/bin/bash

# Python 2.7
export CONDA_PY=27
binstar remove bashtage/arch/1.0/linux-64/arch-1.0-np18py27_0.tar.bz2 --force
conda build binstar

# Python 3.3
export CONDA_PY=33
binstar remove bashtage/arch/1.0/linux-64/arch-1.0-np18py33_0.tar.bz2 --force
conda build binstar
