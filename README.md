[![Documentation Status](https://readthedocs.org/projects/arch/badge/?version=latest)](https://readthedocs.org/projects/arch/?badge=latest)
[![CI Status](https://travis-ci.org/bashtage/arch.svg?branch=master)](https://travis-ci.org/bashtage/arch)
[![Coverage Status](https://coveralls.io/repos/bashtage/arch/badge.png?branch=master)](https://coveralls.io/r/bashtage/arch?branch=master)


# ARCH

This is a work-in-progress for ARCH and related models, written in Python 
(and Cython)

## What is this repository for?

* Mean models
  * Constant mean
  * Heterogeneous Autoregression (HAR)
  * Autoregression (AR)
  * Zero mean
  * Models with and without exogensou regressors
* Volatility models
  * ARCH
  * GARCH
  * TARCH
  * EGARCH
  * EWMA/RiskMetrics
* Distributions
  * Normal
  * Student's T

## Examples

See the [example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/master/examples/examples.ipynb) for a more complete overview.

```python
import datetime as dt
import pandas.io.data as web
st = dt.datetime(1990,1,1)
en = dt.datetime(2014,1,1)
data = web.get_data_yahoo('^FTSE', start=st, end=en)
returns = 100 * data['Adj Close'].pct_change().dropna()

from arch import arch_model
am = arch_model(returns)
res = am.fit()
```

## Documentation
Documentation is hosted on [read the docs](http://arch.readthedocs.org/en/latest/)
 
## Requirements

* NumPy (1.7+)
* SciPy (0.12+)
* Pandas (0.14+)
* statsmodels (0.5+)
* matplotlib (1.3+)

Installing
* Cython (0.20+)
* nose (For tests)
* sphinx (to build docs)
* sphinx-napoleon (to build docs)

## Installing

Setup does not verify requirements.  Please ensure these are installed.

### Linux/OSX

```
pip install git+git://github.com/bashtage/arch.git
```

**Anaconda**

_Anaconda builds are not currently available for OSX._

```
conda install -c https://conda.binstar.org/bashtage arch
```

### Windows

**With a compiler**

If you are comfortable compiling binaries on Windows:

```
pip install git+git://github.com/bashtage/arch.git
```

**No Compiler**

All binary code is backed by a pure Python implementation.  Compiling can be 
skipped using the flag `--no-binary`
 
```
pip install git+git://github.com/bashtage/arch.git --install-option "--no-binary"
```

_Note that it isn't possible to run the test suite will fail if installed with_ `--no-binary`

**Anaconda**

```
conda install -c https://conda.binstar.org/bashtage arch
```

## More about ARCH
More information about ARCH and related models is available in the notes and 
research available at [Kevin Sheppard's site](http://www.kevinsheppard.com).

## Contributing

Contributions are welcome.  There are opportunities at many levels to 
contribute:

* Implement new volatility process, e.g FIGARCH
* Improve docstrings where unclear or with typos
* Provide examples, preferrablly in the form of IPython notebooks


