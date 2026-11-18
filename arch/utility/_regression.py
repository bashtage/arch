"""
OLS with centered regressors, reported for the original regressors.
"""

from typing import Any
import warnings

import numpy as np
from statsmodels.regression.linear_model import (
    OLS,
    OLSResults,
    RegressionResults,
    RegressionResultsWrapper,
)

__all__ = ["centered_ols"]


def centered_ols(endog: Any, exog: Any, **fit_kwargs: Any) -> RegressionResults:
    """
    Estimate OLS with the non-constant regressors centered.

    Parameters
    ----------
    endog : array_like
        The dependent variable.
    exog : array_like
        The regressors.
    **fit_kwargs
        Keyword arguments passed to the fit of the results, for example
        ``cov_type`` and ``cov_kwds``.

    Returns
    -------
    RegressionResults
        The results of the regression of ``endog`` on ``exog``, as returned by
        ``OLS(endog, exog).fit(**fit_kwargs)``.

    Notes
    -----
    If ``exog`` has exactly one constant column and at least one other column, the
    other columns, and ``endog``, are centered on their means before the regression
    is estimated. This only changes the parametrization. The fitted values, less the
    mean of ``endog``, the residuals, the coefficients of the non-constant columns and
    their covariance are those of the regression on ``exog``, and only the constant
    becomes the intercept at the means. The results are transformed back before they
    are returned, so that the coefficient of the constant and its covariance are
    those of the regression on ``exog`` as well.

    A regressor whose variation is small compared with its level, for example the
    lagged level of a series that varies little around a large mean, is nearly a
    multiple of the constant. statsmodels finds the rank from the singular values of
    the regressors, and treats the regression as rank-deficient once the smallest
    singular value is below ``max(nobs, ncols) * eps`` times the largest. The
    estimates are then minimum-norm estimates and not those of the regression.
    Centering removes this collinearity, which comes from the parametrization and
    not from the data.

    If ``exog`` has no constant column, more than one, or no other column, the
    regression is estimated as it is.
    """
    model = OLS(endog, exog)
    regressors = np.asarray(model.exog, dtype=float)
    span = np.ptp(regressors, axis=0)
    constant = np.flatnonzero((span == 0) & (regressors[0] != 0))
    others = np.flatnonzero(span != 0)
    with warnings.catch_warnings():
        # The rank of the regression is checked by the callers that need it
        warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")
        if constant.shape[0] != 1 or others.shape[0] == 0:
            return model.fit(**fit_kwargs)
        const = int(constant[0])
        means = regressors[:, others].mean(axis=0)
        centered = regressors.copy()
        centered[:, others] -= means
        # The dependent variable is centered as well, which only shifts the
        # intercept, so that its level does not reduce the accuracy of the fit
        dependent = np.asarray(model.endog, dtype=float)
        dependent_mean = float(dependent.mean())
        centered_res = OLS(dependent - dependent_mean, centered).fit()
    # centered = regressors @ transform, and so the fitted values are
    # regressors @ (transform @ params of the centered regression) plus the mean
    # of the dependent variable, which is a multiple of the constant column
    level = regressors[0, const]
    transform = np.eye(regressors.shape[1])
    transform[const, others] = -means / level
    params = transform @ np.asarray(centered_res.params)
    params[const] += dependent_mean / level
    cov = transform @ centered_res.normalized_cov_params @ transform.T
    # The attributes that OLS.fit sets on the model, for the original regressors
    model.rank = centered_res.model.rank
    model.pinv_wexog = transform @ centered_res.model.pinv_wexog
    model.normalized_cov_params = cov
    model.wexog_singular_values = np.linalg.svd(model.wexog, compute_uv=False)
    results = OLSResults(model, params, normalized_cov_params=cov, **fit_kwargs)
    return RegressionResultsWrapper(results)
