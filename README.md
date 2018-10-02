# ARCH

Autoregressive Conditional Heteroskedasticity (ARCH) and other tools for
financial econometrics, written in Python (with Cython and/or Numba used
to improve performance)

###### Continuous Integration

[![Travis Build Status](https://travis-ci.org/bashtage/arch.svg?branch=master)](https://travis-ci.org/bashtage/arch)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/nmt02u7jwcgx7i2x?svg=true)](https://ci.appveyor.com/project/bashtage/arch/branch/master)

###### Documentation

[![Documentation Status](https://readthedocs.org/projects/arch/badge/?version=latest)](http://arch.readthedocs.org/en/latest/)

###### Coverage

[![Coverage Status](https://coveralls.io/repos/bashtage/arch/badge.svg?branch=master)](https://coveralls.io/r/bashtage/arch?branch=master)
[![codecov](https://codecov.io/gh/bashtage/arch/branch/master/graph/badge.svg)](https://codecov.io/gh/bashtage/arch)

###### Code Inspections
[![Code Quality: Python](https://img.shields.io/lgtm/grade/python/g/bashtage/arch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/bashtage/arch/context:python)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/bashtage/arch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/bashtage/arch/alerts)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/cea43b588e0f4f2a9d8ba37cf63f8210)](https://www.codacy.com/app/bashtage/arch?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bashtage/arch&amp;utm_campaign=Badge_Grade)
[![codebeat badge](https://codebeat.co/badges/18a78c15-d74b-4820-b56d-72f7e4087532)](https://codebeat.co/projects/github-com-bashtage-arch-master)

###### Citation
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.15681.svg)](http://dx.doi.org/10.5281/zenodo.15681)

## Module Contents

* [Univariate ARCH Models](#volatility)
* [Unit Root Tests](#unit-root)
* [Bootstrapping](#bootstrap)
* [Multiple Comparison Tests](#multiple-comparison)

## Documentation

Released documentation is hosted on
[read the docs](http://arch.readthedocs.org/en/latest/).
Current documentation from the master branch is hosted on
[my github pages](http://bashtage.github.io/arch/doc/index.html).

## More about ARCH

More information about ARCH and related models is available in the notes and
research available at [Kevin Sheppard's site](http://www.kevinsheppard.com).

## Contributing

Contributions are welcome.  There are opportunities at many levels to contribute:

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
  * Generalized Error Distribution

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

* Test of Superior Predictive Ability (SPA), also known as the Reality
  Check or Bootstrap Data Snooper
* Stepwise (StepM)
* Model Confidence Set (MCS)

See the [multiple comparison example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/master/examples/multiple-comparison_examples.ipynb)
for examples of the multiple comparison procedures.

## Requirements

These requirements reflect the testing environment.  It is possible
that arch will work with older versions.

* Python (2.7, 3.5 - 3.7)
* NumPy (1.13+)
* SciPy (0.19+)
* Pandas (0.21+)
* statsmodels (0.8+)
* matplotlib (2.0+)

### Optional Requirements

* Numba (0.35+) will be used if available **and** when installed using
  the --no-binary option
* Jupyter is required to run the notebooks

### Installing

* Cython (0.24+, if not using --no-binary)
* py.test (For tests)
* sphinx (to build docs)
* guzzle_sphinx_theme (to build docs)
* jupyter and notebook (to build docs)
* numpydoc (to build docs)

**Note**: Setup does not verify requirements.  Please ensure these are
installed.

### pip

Releases are available PyPI.

```bash
pip install arch
```

This command should work whether you have a compiler installed or not.
If you want to install with the `--no-binary` options, call

```bash
pip install arch --install-option="--no-binary"
```

You can alternatively install the latest version from GitHub

```bash
pip install git+https://github.com/bashtage/arch.git
```

#### Anaconda

```bash
conda install arch -c bashtage
```

### Windows

Building extension using the community edition of Visual Studio is
well supported for Python 3.5+.  Building extensions for 64-bit Windows
for use in Python 2.7 is also supported using Microsoft Visual C++
Compiler for Python 2.7.  Building on other combinations of Python/Windows
is more difficult and is not necessary when Numba is installed since
just-in-time compiled code (Numba) runs as fast as ahead-of-time
compiled extensions.

#### With a compiler

If you are comfortable compiling binaries on Windows:

```bash
pip install git+https://github.com/bashtage/arch.git
```

#### No Compiler

All binary code is backed by a pure Python implementation.  Compiling
can be skipped using the flag `--no-binary`

```bash
pip install git+https://github.com/bashtage/arch.git --install-option "--no-binary"
```

*Note*: If Cython is not installed, the package will be installed as-if
--no-binary was used.
