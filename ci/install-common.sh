#!/usr/bin/env bash

python -m pip install property_cached flake8 pytest pytest-xdist pytest-cov coverage codacy-coverage coveralls codecov nbformat nbconvert jupyter_client ipython jupyter
if [[ "$DOCBUILD" == true ]]; then
  sudo apt-get install -y enchant
  python -m pip install sphinx ipython numpydoc jupyter seaborn doctr nbsphinx sphinx_material sphinxcontrib-spelling sphinx-autodoc-typehints
fi
