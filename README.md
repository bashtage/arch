[![Documentation Status](https://readthedocs.org/projects/arch/badge/?version=latest)](http://arch.readthedocs.org/en/latest/)
[![CI Status](https://travis-ci.org/bashtage/arch.svg?branch=master)](https://travis-ci.org/bashtage/arch)
[![Coverage Status](https://coveralls.io/repos/bashtage/arch/badge.svg?branch=master)](https://coveralls.io/r/bashtage/arch?branch=master)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.15681.svg)](http://dx.doi.org/10.5281/zenodo.15681)

# ARCH

This is a work-in-progress for ARCH and other tools for financial econometrics, 
written in Python (and Cython)

## What is in this repository?

* [Univariate ARCH Models](#volatility)
* [Unit Root Tests](#unit-root)
* [Bootstrapping](#bootstrap)
* [Multiple Comparison Tests](#multiple-comparison)

## Documentation
Documentation is hosted on [read the docs](http://arch.readthedocs.org/en/latest/)

## More about ARCH
More information about ARCH and related models is available in the notes and 
research available at [Kevin Sheppard's site](http://www.kevinsheppard.com).

## Contributing

Contributions are welcome.  There are opportunities at many levels to 
contribute:

* Implement new volatility process, e.g FIGARCH
* Improve docstrings where unclear or with typos
* Provide examples, preferably in the form of IPython notebooks

## Examples

<a name="volatility"/>
### Volatility Modeling 

* Mean models
    * Constant mean
    * Heterogeneous Autoregression (HAR)
    * Autoregression (AR)
    * Zero mean
    * Models with and without exogenous regressors
* Volatility models
    * ARCH
    * GARCH
    * TARCH
    * EGARCH
    * EWMA/RiskMetrics
* Distributions
    * Normal
    * Student's T

See the [univariate volatility example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/master/examples/univariate_volatility_modeling.ipynb) for a more complete overview.

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

<a name="unit-root"/>
### Unit Root Tests

* Augmented Dickey-Fuller
* Dickey-Fuller GLS
* Phillips-Perron
* KPSS
* Variance Ratio tests

See the [unit root testing example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/master/examples/unitroot_examples.ipynb) for examples of testing series for unit roots.

<a name="bootstrap"/>
### Bootstrap

* Bootstraps
    * IID Bootstrap
    * Stationary Bootstrap
    * Circular Block Bootstrap
    * Moving Block Bootstrap
* Methods
    * Confidence interval construction
    * Covariance estimation
    * Apply method to estimate model across bootstraps
    * Generic Bootstrap iterator

See the [bootstrap example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/master/examples/bootstrap_examples.ipynb) 
for examples of bootstrapping the Sharpe ratio and a Probit model from 
Statsmodels.


```python
# Import data
import datetime as dt
import pandas as pd
import pandas.io.data as web
start = dt.datetime(1951,1,1)
end = dt.datetime(2014,1,1)
sp500 = web.get_data_yahoo('^GSPC', start=start, end=end)
start = sp500.index.min()
end = sp500.index.max()
monthly_dates = pd.date_range(start, end, freq='M')
monthly = sp500.reindex(monthly_dates, method='ffill')
returns = 100 * monthly['Adj Close'].pct_change().dropna()

# Function to compute parameters
def sharpe_ratio(x):
    mu, sigma = 12 * x.mean(), np.sqrt(12 * x.var())
    return np.array([mu, sigma, mu / sigma])

# Bootstrap confidence intervals
from arch.bootstrap import IIDBootstrap
bs = IIDBootstrap(returns)
ci = bs.conf_int(sharpe_ratio, 1000, method='percentile')    
```

<a name="multiple-comparison"/>
### Multiple Comparison Procedures

* Test of Superior Predictive Ability (SPA), also known as the Reality Check or Bootstrap Data Snooper
* Stepwise (StepM)
* Model Confidence Set (MCS)

See the [multiple comparison example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/master/examples/multiple-comparison_examples.ipynb) 
for examples of the multiple comparison procedures.

## Requirements

* NumPy (1.7+)
* SciPy (0.12+)
* Pandas (0.14+)
* statsmodels (0.5+)
* matplotlib (1.3+)

### Optional Requirements

* Numba (0.15+) will be used if available **and** when installed using the 
--no-binary option
* IPython (3.0+) is required to run the notebooks

### Installing

* Cython (0.20+, if not using --no-binary)
* nose (For tests)
* sphinx (to build docs)
* sphinx-napoleon (to build docs)

**Note**: Setup does not verify requirements.  Please ensure these are installed.

### Linux/OSX

```
pip install git+https://github.com/bashtage/arch.git
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
pip install git+https://github.com/bashtage/arch.git
```

**No Compiler**

All binary code is backed by a pure Python implementation.  Compiling can be 
skipped using the flag `--no-binary`
 
```
pip install git+https://github.com/bashtage/arch.git --install-option "--no-binary"
```

_Note: the test suite compares the Numba implementations against Cython 
implementations of some recursions, and so it is not possible to run the 
test suite when installing with_ `--no-binary` .

**Anaconda**

```
conda install -c https://conda.binstar.org/bashtage arch
```

