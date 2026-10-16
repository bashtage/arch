# arch

[![arch](https://bashtage.github.io/arch/doc/_static/images/color-logo-256.png)](https://github.com/bashtage/arch)

Autoregressive Conditional Heteroskedasticity (ARCH) and other tools for
financial econometrics, written in Python (with Cython and/or Numba used
to improve performance)

| Metric                     |                                                                                                                                                                                                                                          |
| :------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Latest Release**         | [![PyPI version](https://badge.fury.io/py/arch.svg)](https://badge.fury.io/py/arch)                                                                                                                                                      |
|                            | [![conda-forge version](https://anaconda.org/conda-forge/arch-py/badges/version.svg)](https://anaconda.org/conda-forge/arch-py)                                                                                                          |
| **Continuous Integration** | [![Build Status](https://dev.azure.com/kevinksheppard0207/kevinksheppard/_apis/build/status/bashtage.arch?branchName=main)](https://dev.azure.com/kevinksheppard0207/kevinksheppard/_build/latest?definitionId=1&branchName=main)        |
| **Coverage**               | [![codecov](https://codecov.io/gh/bashtage/arch/branch/main/graph/badge.svg)](https://codecov.io/gh/bashtage/arch)                                                                                                                       |
| **Code Quality**           | [![Codacy Badge](https://api.codacy.com/project/badge/Grade/93f6fd90209842bf97fd20fda8db70ef)](https://www.codacy.com/manual/bashtage/arch?utm_source=github.com&utm_medium=referral&utm_content=bashtage/arch&utm_campaign=Badge_Grade) |
| **Citation**               | [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.593254.svg)](https://doi.org/10.5281/zenodo.593254)                                                                                                                                  |
| **Documentation**          | [![Documentation](https://github.com/bashtage/arch/actions/workflows/generate-documentation.yml/badge.svg)](https://bashtage.github.io/arch/)                                                                                                          |

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

Documentation for the latest release is hosted on
[GitHub Pages](https://bashtage.github.io/arch/), and documentation from the
main branch is at [bashtage.github.io/arch/devel](https://bashtage.github.io/arch/devel/).

## More about ARCH

More information about ARCH and related models is available in the notes and
research available at [Kevin Sheppard's site](https://www.kevinsheppard.com).

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

See the [univariate volatility example notebook](https://bashtage.github.io/arch/univariate/univariate_volatility_modeling.html) for a more complete overview.

```python
import arch.data.sp500
data = arch.data.sp500.load()
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

See the [unit root testing example notebook](https://bashtage.github.io/arch/unitroot/unitroot_examples.html)
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

See the [cointegration testing example notebook](https://bashtage.github.io/arch/unitroot/unitroot_cointegration_examples.html)
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

See the [bootstrap example notebook](https://bashtage.github.io/arch/bootstrap/bootstrap_examples.html)
for examples of bootstrapping the Sharpe ratio and a Probit model from statsmodels.

```python
# Import data
import pandas as pd
import numpy as np
import arch.data.sp500
sp500 = arch.data.sp500.load()
start = sp500.index.min()
end = sp500.index.max()
monthly_dates = pd.date_range(start, end, freq='ME')
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

See the [multiple comparison example notebook](https://bashtage.github.io/arch/multiple-comparison/multiple-comparison_examples.html)
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

- Python (3.10-3.15)
- NumPy (1.22.3+)
- SciPy (1.8+)
- pandas (1.4+)
- statsmodels (0.13.1+)
- matplotlib (3+), optional

arch 9.0 is the last release that supports Python 3.10 and NumPy 1.x. Later
releases will raise the other minimum versions accordingly.


### Optional Requirements

- Numba (0.55+) will be used if available **and** when arch is installed without building the binary modules. To make sure that they are not built, install from the source distribution with the `no-binary` build option:

```shell
python -m pip install arch --no-binary arch -Csetup-args="-Dno-binary=true"
```

- jupyter and notebook are required to run the notebooks

## Installing

Wheels are available for Linux, macOS and Windows, so installing with
`pip` does not need a compiler. Building from source needs a C compiler;
`pip` installs the build requirements (Cython, meson-python and NumPy)
automatically.

### pip

Releases are available PyPI and can be installed with `pip`.

```shell
pip install arch
```

You can alternatively install the latest version from GitHub

```bash
pip install git+https://github.com/bashtage/arch.git
```

Adding `-Csetup-args="-Dno-binary=true"` to the `pip install` command
disables compilation of the extensions.

### Anaconda

`conda` users can install from conda-forge,

```bash
conda install arch-py -c conda-forge
```

**Note**: The conda-forge name is `arch-py`.

### Windows

Building the extensions from source needs the Visual Studio Build Tools
(the free community edition is enough).

### Developing

The development requirements are:

- Cython (3.1+), meson-python and ninja, unless building without the binary
  modules
- pytest (9+, for tests)
- sphinx (to build docs)
- sphinx-immaterial (to build docs)
- jupyter, notebook and nbsphinx (to build docs)

### Installation Notes

1. Building without the binary modules (`-Csetup-args="-Dno-binary=true"`)
   still needs a C compiler and Cython, since the build checks for them.
   The recursions then run in numba if it is installed, and in pure
   Python otherwise, which is much slower.
