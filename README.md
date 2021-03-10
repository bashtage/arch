# arch

[![arch](https://bashtage.github.io/arch/doc/_static/images/color-logo-256.png)](https://github.com/bashtage/arch)

Autoregressive Conditional Heteroskedasticity (ARCH) and other tools for
financial econometrics, written in Python (with Cython and/or Numba used
to improve performance)

| Metric                     |                                                                                                                                                                                                                                          |
| :------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Latest Release**         | [![PyPI version](https://badge.fury.io/py/arch.svg)](https://badge.fury.io/py/arch)                                                                                                                                                      |
|                            | [![conda-forge version](https://anaconda.org/conda-forge/arch-py/badges/version.svg)](https://anaconda.org/conda-forge/arch-py)                                                                                          |
| **Continuous Integration** | [![Build Status](https://dev.azure.com/kevinksheppard0207/kevinksheppard/_apis/build/status/bashtage.arch?branchName=main)](https://dev.azure.com/kevinksheppard0207/kevinksheppard/_build/latest?definitionId=1&branchName=main)        |
|                            | [![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/nmt02u7jwcgx7i2x?svg=true)](https://ci.appveyor.com/project/bashtage/arch/branch/main)                                                                             |
| **Coverage**               | [![codecov](https://codecov.io/gh/bashtage/arch/branch/main/graph/badge.svg)](https://codecov.io/gh/bashtage/arch)                                                                                                                       |
| **Code Quality**           | [![Code Quality: Python](https://img.shields.io/lgtm/grade/python/g/bashtage/arch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/bashtage/arch/context:python)                                                                 |
|                            | [![Total Alerts](https://img.shields.io/lgtm/alerts/g/bashtage/arch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/bashtage/arch/alerts)                                                                                       |
|                            | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/93f6fd90209842bf97fd20fda8db70ef)](https://www.codacy.com/manual/bashtage/arch?utm_source=github.com&utm_medium=referral&utm_content=bashtage/arch&utm_campaign=Badge_Grade) |
|                            | [![codebeat badge](https://codebeat.co/badges/18a78c15-d74b-4820-b56d-72f7e4087532)](https://codebeat.co/projects/github-com-bashtage-arch-main)                                                                                         |
| **Citation**               | [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.593254.svg)](https://doi.org/10.5281/zenodo.593254)                                                                                                                                  |
| **Documentation**          | [![Documentation Status](https://readthedocs.org/projects/arch/badge/?version=latest)](http://arch.readthedocs.org/en/latest/)                                                                                                           |

## Module Contents

- [Univariate ARCH Models](#volatility)
- [Unit Root Tests](#unit-root)
- [Cointegration Testing and Analysis](#cointegration)
- [Bootstrapping](#bootstrap)
- [Multiple Comparison Tests](#multiple-comparison)
- [Long-run Covariance Estimation](#long-run-covariance)

### Python 3

`arch` is Python 3 only. Version 4.8 is the final version that supported Python 2.7.

## Documentation

Released documentation is hosted on
[read the docs](http://arch.readthedocs.org/en/latest/).
Current documentation from the main branch is hosted on
[my github pages](http://bashtage.github.io/arch/doc/index.html).

## More about ARCH

More information about ARCH and related models is available in the notes and
research available at [Kevin Sheppard's site](http://www.kevinsheppard.com).

## Contributing

Contributions are welcome. There are opportunities at many levels to contribute:

- Implement new volatility process, e.g., FIGARCH
- Improve docstrings where unclear or with typos
- Provide examples, preferably in the form of IPython notebooks

## Examples

<a id="volatility"></a>

### Volatility Modeling

- Mean models
  - Constant mean
  - Heterogeneous Autoregression (HAR)
  - Autoregression (AR)
  - Zero mean
  - Models with and without exogenous regressors
- Volatility models
  - ARCH
  - GARCH
  - TARCH
  - EGARCH
  - EWMA/RiskMetrics
- Distributions
  - Normal
  - Student's T
  - Generalized Error Distribution

See the [univariate volatility example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/main/examples/univariate_volatility_modeling.ipynb) for a more complete overview.

```python
import datetime as dt
import pandas_datareader.data as web
st = dt.datetime(1990,1,1)
en = dt.datetime(2014,1,1)
data = web.get_data_yahoo('^FTSE', start=st, end=en)
returns = 100 * data['Adj Close'].pct_change().dropna()

from arch import arch_model
am = arch_model(returns)
res = am.fit()
```

<a id="unit-root"></a>

### Unit Root Tests

- Augmented Dickey-Fuller
- Dickey-Fuller GLS
- Phillips-Perron
- KPSS
- Zivot-Andrews
- Variance Ratio tests

See the [unit root testing example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/main/examples/unitroot_examples.ipynb)
for examples of testing series for unit roots.

<a id="unit-root"></a>

### Cointegration Testing and Analysis

- Tests
  - Engle-Granger Test
  - Phillips-Ouliaris Test
- Cointegration Vector Estimation
  - Canonical Cointegrating Regression 
  - Dynamic OLS
  - Fully Modified OLS

See the [cointegration testing example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/main/examples/unitroot_cointegration_examples.ipynb)
for examples of testing series for cointegration.

<a id="bootstrap"></a>

### Bootstrap

- Bootstraps
  - IID Bootstrap
  - Stationary Bootstrap
  - Circular Block Bootstrap
  - Moving Block Bootstrap
- Methods
  - Confidence interval construction
  - Covariance estimation
  - Apply method to estimate model across bootstraps
  - Generic Bootstrap iterator

See the [bootstrap example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/main/examples/bootstrap_examples.ipynb)
for examples of bootstrapping the Sharpe ratio and a Probit model from statsmodels.

```python
# Import data
import datetime as dt
import pandas as pd
import numpy as np
import pandas_datareader.data as web
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

<a id="multiple-comparison"></a>

### Multiple Comparison Procedures

- Test of Superior Predictive Ability (SPA), also known as the Reality
  Check or Bootstrap Data Snooper
- Stepwise (StepM)
- Model Confidence Set (MCS)

See the [multiple comparison example notebook](http://nbviewer.ipython.org/github/bashtage/arch/blob/main/examples/multiple-comparison_examples.ipynb)
for examples of the multiple comparison procedures.


<a id="long-run-covariance"></a>

### Long-run Covariance Estimation

Kernel-based estimators of long-run covariance including the
Bartlett kernel which is known as Newey-West in econometrics.
Automatic bandwidth selection is available for all of the 
covariance estimators.

```python
from arch.covariance.kernel import Bartlett
from arch.data import nasdaq
data = nasdaq.load()
returns = data[["Adj Close"]].pct_change().dropna()

cov_est = Bartlett(returns ** 2)
# Get the long-run covariance
cov_est.cov.long_run
```

## Requirements

These requirements reflect the testing environment. It is possible
that arch will work with older versions.

- Python (3.7+)
- NumPy (1.16+)
- SciPy (1.2+)
- Pandas (0.23+)
- statsmodels (0.11+)
- matplotlib (2.2+), optional
- property-cached (1.6.4+), optional

### Optional Requirements

- Numba (0.35+) will be used if available **and** when installed using the --no-binary option
- jupyter and notebook are required to run the notebooks

## Installing

Standard installation with a compiler requires Cython. If you do not
have a compiler installed, the `arch` should still install. You will
see a warning but this can be ignored. If you don't have a compiler,
`numba` is strongly recommended.

### pip

Releases are available PyPI and can be installed with `pip`.

```bash
pip install arch
```

This command should work whether you have a compiler installed or not.
If you want to install with the `--no-binary` options, use

```bash
pip install arch --install-option="--no-binary" --no-build-isoloation
```

The `--no-build-isoloation` uses the existing NumPy when building the
source. This is usually needed since pip will attempt to build all
dependencies from source when `--install-option` is used.

You can alternatively install the latest version from GitHub

```bash
pip install git+https://github.com/bashtage/arch.git
```

`--install-option="--no-binary" --no-build-isoloation` can be used to
disable compilation of the extensions.

### Anaconda

`conda` users can install from conda-forge,

```bash
conda install arch-py -c conda-forge
```

**Note**: The conda-forge name is `arch-py`.

### Windows

Building extension using the community edition of Visual Studio is
well supported for Python 3.6+. Building on other combinations of
Python/Windows is more difficult and is not necessary when numba
is installed since just-in-time compiled code (numba) runs as fast as
ahead-of-time compiled extensions.

### Developing

The development requirements are:

- Cython (0.29+, if not using --no-binary)
- pytest (For tests)
- sphinx (to build docs)
- sphinx_material (to build docs)
- jupyter, notebook and nbsphinx (to build docs)

### Installation Notes

1. If Cython is not installed, the package will be installed
   as-if `--no-binary` was used.
2. Setup does not verify these requirements. Please ensure these are
   installed.
