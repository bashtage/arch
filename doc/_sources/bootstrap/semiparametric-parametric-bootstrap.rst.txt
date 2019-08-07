.. _semiparametric-bootstraps:

Semiparametric Bootstraps
-------------------------

Functions for semi-parametric bootstraps differ from those used in
nonparametric bootstraps.  At a minimum they must accept the keyword
argument ``params`` which will contain the parameters estimated on
the original (non-bootstrap) data.  This keyword argument must be
optional so that the function can be called without the keyword
argument to estimate parameters.  In most applications other inputs
will also be needed to perform the semi-parametric step - these can
be input using the ``extra_kwargs`` keyword input.

For simplicity, consider a semiparametric bootstrap of an OLS regression.
The bootstrap step will combine the original parameter estimates and original
regressors with bootstrapped residuals to construct a bootstrapped
regressand.  The bootstrap regressand and regressors can then be used to
produce a bootstrapped parameter estimate.

The user-provided function must:

- Estimate the parameters when ``params`` is not provided
- Estimate residuals from bootstrapped data when ``params`` is provided
  to construct bootstrapped residuals, simulate the regressand, and then
  estimate the bootstrapped parameters

.. code-block:: python

    import numpy as np
    def ols(y, x, params=None, x_orig=None):
        if params is None:
            return np.linalg.pinv(x).dot(y).ravel()

        # When params is not None
        # Bootstrap residuals
        resids = y - x.dot(params)
        # Simulated data
        y_star = x_orig.dot(params) + resids
        # Parameter estimates
        return np.linalg.pinv(x_orig).dot(y_star).ravel()


.. note::

  The function should return a 1-dimensional array. ``ravel`` is used above to
  ensure that the parameters estimated are 1d.

This function can then be used to perform a semiparametric bootstrap

.. code-block:: python

    from arch.bootstrap import IIDBootstrap
    x = np.random.randn(100, 3)
    e = np.random.randn(100, 1)
    b = np.arange(1, 4)[:, None]
    y = x.dot(b) + e
    bs = IIDBootstrap(y, x)
    ci = bs.conf_int(ols, 1000, method='percentile',
                     sampling='semi', extra_kwargs={'x_orig': x})

Using ``partial`` instead of ``extra_kwargs``
=============================================

``functools.partial`` can be used instead to provide a wrapper function which
can then be used in the bootstrap.  This example fixed the value of ``x_orig``
so that it is not necessary to use ``extra_kwargs``.

.. code-block:: python

    from functools import partial
    ols_partial = partial(ols, x_orig=x)
    ci = bs.conf_int(ols_partial, 1000, sampling='semi')

Semiparametric Bootstrap (Alternative Method)
=============================================

Since semiparametric bootstraps are effectively bootstrapping residuals, an
alternative method can be used to conduct a semiparametric bootstrap. This
requires passing both the data and the estimated residuals when initializing
the bootstrap.

First, the function used must be account for this structure.

.. code-block:: python

    def ols_semi_v2(y, x, resids=None, params=None, x_orig=None):
        if params is None:
            return np.linalg.pinv(x).dot(y).ravel()

        # Simulated data if params provided
        y_star = x_orig.dot(params) + resids
        # Parameter estimates
        return np.linalg.pinv(x_orig).dot(y_star).ravel()

This version can then be used to *directly* implement a semiparametric
bootstrap, although ultimately it is not meaningfully simpler than the
previous method.

.. code-block:: python

    resids = y - x.dot(ols_semi_v2(y,x))
    bs = IIDBootstrap(y, x, resids=resids)
    bs.conf_int(ols_semi_v2, 1000, sampling='semi', extra_kwargs={'x_orig': x})

.. note::

    This alternative method is more useful when computing residuals is
    relatively expensive when compared to simulating data or estimating
    parameters.  These circumstances are rarely encountered in actual problems.

.. _parametric-bootstraps:

Parametric Bootstraps
---------------------

Parametric bootstraps are meaningfully different from their nonparametric or
semiparametric cousins.  Instead of sampling the data to simulate the data
(or residuals, in the case of a semiparametric bootstrap), a parametric
bootstrap makes use of a fully parametric model to simulate data using a
pseudo-random number generator.

.. warning::

    Parametric bootstraps are model-based methods to construct exact
    confidence intervals through integration.   Since these confidence
    intervals should be exact, bootstrap methods which make use of
    asymptotic normality are required (and may not be desirable).

Implementing a parametric bootstrap, like implementing a semi-parametric
bootstrap, requires specific keyword arguments. The first is ``params``,
which, when present, will contain the parameters estimated on the original
data.  The second is ``rng`` which will contain the
:class:`numpy.random.RandomState` instance that is used by the bootstrap.
This is provided to facilitate simulation in a reproducible manner.

A parametric bootstrap function must:

- Estimate the parameters when ``params`` is not provided
- Simulate data when ``params`` is provided and then
  estimate the bootstrapped parameters on the simulated data

This example continues the OLS example from the semiparametric example,
only assuming that residuals are normally distributed.  The variance
estimator is the MLE.

.. code-block:: python

    def ols_para(y, x, params=None, state=None, x_orig=None):
        if params is None:
            beta = np.linalg.pinv(x).dot(y)
            e = y - x.dot(beta)
            sigma2 = e.T.dot(e) / e.shape[0]
            return np.r_[beta.ravel(), sigma2.ravel()]

        beta = params[:-1]
        sigma2 = params[-1]
        e = state.standard_normal(x_orig.shape[0])
        ystar = x_orig.dot(beta) + np.sqrt(sigma2) * e

        # Use the plain function to compute parameters
        return ols_para(ystar, x_orig)

This function can then be used to form parametric bootstrap confidence intervals.

.. code-block:: python

    bs = IIDBootstrap(y,x)
    ci = bs.conf_int(ols_para, 1000, method='percentile',
                     sampling='parametric', extra_kwargs={'x_orig': x})

.. note::

    The parameter vector in this example includes the variance since this is
    required when specifying a complete model.
