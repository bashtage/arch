from typing import cast
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.regression.linear_model import RegressionResults

from arch._typing import ArrayLike1D, ArrayLike2D, Literal, UnitRootTrend
from arch.covariance.kernel import CovarianceEstimator
from arch.unitroot._shared import (
    KERNEL_ERR,
    KERNEL_ESTIMATORS,
    ResidualCointegrationTestResult,
    _cross_section,
)
from arch.unitroot.critical_values.phillips_ouliaris import (
    CV_PARAMETERS,
    CV_TAU_MIN,
    PVAL_LARGE_P,
    PVAL_SMALL_P,
    PVAL_TAU_MAX,
    PVAL_TAU_MIN,
    PVAL_TAU_STAR,
)
from arch.unitroot.unitroot import TREND_DESCRIPTION
from arch.utility.array import ensure2d
from arch.utility.io import str_format
from arch.utility.timeseries import add_trend


class CriticalValueWarning(RuntimeWarning):
    pass


def _po_ptests(
    z: pd.DataFrame,
    xsection: RegressionResults,
    test_type: Literal["Pu", "Pz"],
    trend: UnitRootTrend,
    kernel: str,
    bandwidth: int | None,
    force_int: bool,
) -> "PhillipsOuliarisTestResults":
    nobs = z.shape[0]
    z_lead = z.iloc[1:]
    z_lag = add_trend(z.iloc[:-1], trend=trend)
    phi = np.linalg.lstsq(z_lag, z_lead, rcond=None)[0]
    xi = z_lead - np.asarray(z_lag @ phi)

    ker_est = KERNEL_ESTIMATORS[kernel]
    cov_est = ker_est(xi, bandwidth=bandwidth, center=False, force_int=force_int)
    cov = cov_est.cov
    # Rescale to match definition in PO
    omega = (nobs - 1) / nobs * np.asarray(cov.long_run)

    u = np.asarray(xsection.resid)
    if test_type.lower() == "pu":
        denom = u.T @ u / nobs
        omega21 = omega[0, 1:]
        omega22 = omega[1:, 1:]
        omega22_inv = np.linalg.inv(omega22)
        omega112 = omega[0, 0] - np.squeeze(omega21.T @ omega22_inv @ omega21)
        test_stat = nobs * float(np.squeeze(omega112 / denom))
    else:
        # returning p_z
        _z = np.asarray(z)
        if trend != "n":
            tr = add_trend(nobs=_z.shape[0], trend=trend)
            _z = _z - tr @ np.linalg.lstsq(tr, _z, rcond=None)[0]
        else:
            _z = _z - _z[:1]  # Ensure first observation is 0
        m_zz = _z.T @ _z / nobs
        test_stat = nobs * float(np.squeeze((omega @ np.linalg.inv(m_zz)).trace()))
    cv = phillips_ouliaris_cv(test_type, trend, z.shape[1], z.shape[0])
    pval = phillips_ouliaris_pval(test_stat, test_type, trend, z.shape[1])
    return PhillipsOuliarisTestResults(
        test_stat,
        pval,
        cv,
        order=z.shape[1],
        xsection=xsection,
        test_type=test_type,
        kernel_est=cov_est,
    )


def _po_ztests(
    yx: pd.DataFrame,
    xsection: RegressionResults,
    test_type: Literal["Za", "Zt"],
    trend: UnitRootTrend,
    kernel: str,
    bandwidth: int | None,
    force_int: bool,
) -> "PhillipsOuliarisTestResults":
    # Za and Zt tests
    u = np.asarray(xsection.resid)[:, None]
    nobs = u.shape[0]
    # Rescale to match definition in PO
    k_scale = (nobs - 1) / nobs
    alpha = np.linalg.lstsq(u[:-1], u[1:, 0], rcond=None)[0]
    k = u[1:] - alpha * u[:-1]
    u2 = np.squeeze(u[:-1].T @ u[:-1])
    kern_est = KERNEL_ESTIMATORS[kernel]
    cov_est = kern_est(k, bandwidth=bandwidth, center=False, force_int=force_int)
    cov = cov_est.cov
    one_sided_strict = k_scale * cov.one_sided_strict

    z = float(np.squeeze((alpha - 1) - nobs * one_sided_strict / u2))
    if test_type.lower() == "za":
        test_stat = nobs * z
    else:
        long_run = k_scale * np.squeeze(cov.long_run)
        avar = long_run / u2
        se = np.sqrt(avar)
        test_stat = z / se
    cv = phillips_ouliaris_cv(test_type, trend, yx.shape[1], yx.shape[0])
    pval = phillips_ouliaris_pval(test_stat, test_type, trend, yx.shape[1])
    x = xsection.model.exog
    return PhillipsOuliarisTestResults(
        test_stat,
        pval,
        cv,
        order=x.shape[1] + 1,
        xsection=xsection,
        test_type=test_type,
        kernel_est=cov_est,
    )


def phillips_ouliaris(
    y: ArrayLike1D,
    x: ArrayLike2D,
    trend: UnitRootTrend = "c",
    *,
    test_type: Literal["Za", "Zt", "Pu", "Pz"] = "Zt",
    kernel: str = "bartlett",
    bandwidth: int | None = None,
    force_int: bool = False,
) -> "PhillipsOuliarisTestResults":
    r"""
    Test for cointegration within a set of time series.

    Parameters
    ----------
    y : array_like
        The left-hand-side variable in the cointegrating regression.
    x : array_like
        The right-hand-side variables in the cointegrating regression.
    trend : {"n","c","ct","ctt"}, default "c"
        Trend to include in the cointegrating regression. Trends are:

        * "n": No deterministic terms
        * "c": Constant
        * "ct": Constant and linear trend
        * "ctt": Constant, linear and quadratic trends
    test_type : {"Za", "Zt", "Pu", "Pz"}, default "Zt"
        The test statistic to compute. Supported options are:

        * "Za": The Zα test based on the the debiased AR(1) coefficient.
        * "Zt": The Zt test based on the t-statistic from an AR(1).
        * "Pu": The Pᵤ variance-ratio test.
        * "Pz": The Pz test of the trace of the product of an estimate of the
          long-run residual variance and the inner-product of the data.

        See the notes for details on the test.
    kernel : str, default "bartlett"
        The string name of any of any known kernel-based long-run
        covariance estimators. Common choices are "bartlett" for the
        Bartlett kernel (Newey-West), "parzen" for the Parzen kernel
        and "quadratic-spectral" for the Quadratic Spectral kernel.
    bandwidth : int, default None
        The bandwidth to use. If not provided, the optimal bandwidth is
        estimated from the data. Setting the bandwidth to 0 produces White's
        covariance estimator.
    force_int : bool, default False
        Whether the force the estimated optimal bandwidth to be an integer.

    Returns
    -------
    PhillipsOuliarisTestResults
        Results of the Phillips-Ouliaris test.

    See Also
    --------
    arch.unitroot.ADF
        Augmented Dickey-Fuller testing.
    arch.unitroot.PhillipsPerron
        Phillips & Perron's unit root test.
    arch.unitroot.cointegration.engle_granger
        Engle & Granger's cointegration test.

    Notes
    -----
    Supports 4 distinct tests.

    Define the cross-sectional regression

    .. math::

       y_t = x_t \beta + d_t \gamma + u_t

    where :math:`d_t` are any included deterministic terms. Let
    :math:`\hat{u}_t = y_t - x_t \hat{\beta} + d_t \hat{\gamma}`.

    The Zα and Zt statistics are defined as

    .. math::

       \hat{Z}_\alpha & = T \times z \\
       \hat{Z}_t & =  \frac{\hat{\sigma}_u}{\hat{\omega}^2} \times \sqrt{T} z \\
       z & = (\hat{\alpha} - 1) - \hat{\omega}^2_1 / \hat{\sigma}^2_u

    where :math:`\hat{\sigma}^2_u=T^{-1}\sum_{t=2}^T \hat{u}_t^2`,
    :math:`\hat{\omega}^2_1` is an estimate of the one-sided strict
    autocovariance, and :math:`\hat{\omega}^2` is an estimate of the long-run
    variance of the process.

    The :math:`\hat{P}_u` variance-ratio statistic is defined as

    .. math::

       \hat{P}_u = \frac{\hat{\omega}_{11\cdot2}}{\tilde{\sigma}^2_u}

    where :math:`\tilde{\sigma}^2_u=T^{-1}\sum_{t=1}^T \hat{u}_t^2` and

    .. math::

       \hat{\omega}_{11\cdot 2} = \hat{\omega}_{11}
                                 - \hat{\omega}'_{21} \hat{\Omega}_{22}^{-1}
                                   \hat{\omega}_{21}

    and

    .. math::

       \hat{\Omega}=\left[\begin{array}{cc} \hat{\omega}_{11} & \hat{\omega}'_{21}\\
                                            \hat{\omega}_{21} & \hat{\Omega}_{22}
                    \end{array}\right]

    is an estimate of the long-run covariance of :math:`\xi_t`, the residuals
    from an VAR(1) on :math:`z_t=[y_t,z_t]` that includes and trends included
    in the test.

    .. math::

       z_t = \Phi z_{t-1} + \xi_\tau

    The final test statistic is defined

    .. math::

       \hat{P}_z = T \times \mathrm{tr}(\hat{\Omega} M_{zz}^{-1})

    where :math:`M_{zz} = \sum_{t=1}^T \tilde{z}'_t \tilde{z}_t`,
    :math:`\tilde{z}_t` is the vector of data :math:`z_t=[y_t,x_t]` detrended
    using any trend terms included in the test,
    :math:`\tilde{z}_t = z_t - d_t \hat{\kappa}` and :math:`\hat{\Omega}` is
    defined above.

    The specification of the :math:`\hat{P}_z` test statistic when trend is "n"
    differs from the expression in [1]_. We recenter :math:`z_t` by subtracting
    the first observation, so that :math:`\tilde{z}_t = z_t - z_1`. This is
    needed to ensure that the initial value does not affect the distribution
    under the null. When the trend is anything other than "n", this step is not
    needed and the test statistics is identical whether the first observation
    is subtracted or not.

    References
    ----------
    .. [1] Phillips, P. C., & Ouliaris, S. (1990). Asymptotic properties of
       residual based tests for cointegration. Econometrica: Journal of the
       Econometric Society, 165-193.
    """
    test_type_key = test_type.lower()
    if test_type_key not in ("za", "zt", "pu", "pz"):
        raise ValueError(
            f"Unknown test_type: {test_type}. Only Za, Zt, Pu and Pz are supported."
        )
    kernel = kernel.lower().replace("-", "").replace("_", "")
    if kernel not in KERNEL_ESTIMATORS:
        raise ValueError(KERNEL_ERR)
    y_2d = ensure2d(y, "y")
    x = ensure2d(x, "x")
    xsection = _cross_section(y_2d, x, trend)
    data = xsection.model.data
    x_df = data.orig_exog.iloc[:, : x.shape[1]]
    z = pd.concat([data.orig_endog, x_df], axis=1)
    if test_type_key in ("pu", "pz"):
        return _po_ptests(
            z,
            xsection,
            cast("Literal['Pu', 'Pz']", test_type),
            trend,
            kernel,
            bandwidth,
            force_int,
        )
    return _po_ztests(
        z,
        xsection,
        cast("Literal['Za', 'Zt']", test_type),
        trend,
        kernel,
        bandwidth,
        force_int,
    )


class PhillipsOuliarisTestResults(ResidualCointegrationTestResult):
    def __init__(
        self,
        stat: float,
        pvalue: float,
        crit_vals: pd.Series,
        null: str = "No Cointegration",
        alternative: str = "Cointegration",
        trend: str = "c",
        order: int = 2,
        xsection: RegressionResults | None = None,
        test_type: str = "Za",
        kernel_est: CovarianceEstimator | None = None,
        rho: float = 0.0,
    ) -> None:
        super().__init__(
            stat, pvalue, crit_vals, null, alternative, trend, order, xsection=xsection
        )
        self.name = f"Phillips-Ouliaris {test_type} Cointegration Test"
        self._test_type = test_type
        assert kernel_est is not None
        self._kernel_est = kernel_est
        self._rho = rho
        self._additional_info.update(
            {
                "Kernel": self.kernel,
                "Bandwidth": str_format(kernel_est.bandwidth),
                "Trend": self.trend,
                "Distribution Order": self.distribution_order,
            }
        )

    @property
    def kernel(self) -> str:
        """Name of the long-run covariance estimator"""
        return self._kernel_est.__class__.__name__

    @property
    def bandwidth(self) -> float:
        """Bandwidth used by the long-run covariance estimator"""
        return self._kernel_est.bandwidth

    def summary(self) -> Summary:
        """Summary of test, containing statistic, p-value and critical values"""
        if self.bandwidth == int(self.bandwidth):
            bw = str(int(self.bandwidth))
        else:
            bw = f"{self.bandwidth:0.3f}"
        table_data = [
            ("Test Statistic", f"{self.stat:0.3f}"),
            ("P-value", f"{self.pvalue:0.3f}"),
            ("Kernel", f"{self.kernel}"),
            ("Bandwidth", bw),
        ]
        title = self.name

        table = SimpleTable(
            table_data,
            stubs=None,
            title=title,
            colwidths=18,
            datatypes=[0, 1],
            data_aligns=("l", "r"),
        )

        smry = Summary()
        smry.tables.append(table)

        cv_string = "Critical Values: "
        for val in self.critical_values.keys():
            p = str(int(val)) + "%"
            cv_string += f"{self.critical_values[val]:0.2f}"
            cv_string += " (" + p + ")"
            cv_string += ", "
        # Remove trailing ,<space>
        cv_string = cv_string[:-2]

        extra_text = [
            "Trend: " + TREND_DESCRIPTION[self._trend],
            cv_string,
            "Null Hypothesis: " + self.null_hypothesis,
            "Alternative Hypothesis: " + self.alternative_hypothesis,
            "Distribution Order: " + str(self.distribution_order),
        ]

        smry.add_extra_txt(extra_text)
        return smry


def phillips_ouliaris_cv(
    test_type: Literal["Za", "Zt", "Pu", "Pz"],
    trend: UnitRootTrend,
    num: int,
    nobs: int,
) -> pd.Series:
    """
    Critical Values for Phillips-Ouliaris tests

    Parameters
    ----------
    test_type : {"Za", "Zt", "Pu", "Pz"}
        The test type
    trend : {"n", "c", "ct", "ctt"}
        The trend included in the model
    num : int
        Number of assumed stochastic trends in the model under the null. Must
        be between 2 and 13.
    nobs : int
        The number of observations in the time series.

    Returns
    -------
    Series
        The critical values for 1, 5 and 10%
    """
    test_types = ("Za", "Zt", "Pu", "Pz")
    test_type_key = test_type.capitalize()
    if test_type_key not in test_types:
        raise ValueError(f"test_type must be one of: {', '.join(test_types)}")
    trends = ("n", "c", "ct", "ctt")
    if trend not in trends:
        valid = ",".join(trends)
        raise ValueError(f"trend must by one of: {valid}")
    if not 2 <= num <= 13:
        raise ValueError(
            "The number of stochastic trends must be between 2 and 12 (inclusive)"
        )
    key = (test_type_key, trend, num)
    tbl = CV_PARAMETERS[key]
    min_size = CV_TAU_MIN[key]
    if nobs < min_size:

        warnings.warn(
            "The sample size is smaller than the smallest sample size used "
            "to construct the critical value tables. Interpret test "
            "results with caution.",
            CriticalValueWarning,
            stacklevel=2,
        )

    crit_vals = {}
    for size in (10, 5, 1):
        params = tbl[size]
        x = 1.0 / (nobs ** np.arange(4.0))
        crit_vals[size] = x @ params
    return pd.Series(crit_vals)


def phillips_ouliaris_pval(
    stat: float,
    test_type: Literal["Za", "Zt", "Pu", "Pz"],
    trend: UnitRootTrend,
    num: int,
) -> float:
    """
    Asymptotic P-values for Phillips-Ouliaris t-tests

    Parameters
    ----------
    stat : float
        The test statistic
    test_type : {"Za", "Zt", "Pu", "Pz"}
        The test type
    trend : {"n", "c", "ct", "ctt"}
        The trend included in the model
    num : int
        Number of assumed stochastic trends in the model under the null. Must
        be between 2 and 13.

    Returns
    -------
    float
        The asymptotic p-value
    """
    test_types = ("Za", "Zt", "Pu", "Pz")
    test_type_key = test_type.capitalize()
    if test_type_key not in test_types:
        raise ValueError(f"test_type must be one of: {', '.join(test_types)}")
    trends = ("n", "c", "ct", "ctt")
    if trend not in trends:
        valid = ",".join(trends)
        raise ValueError(f"trend must by one of: {valid}")
    if not 2 <= num <= 13:
        raise ValueError(
            "The number of stochastic trends must be between 2 and 12 (inclusive)"
        )
    key = (test_type_key, trend, num)
    if test_type_key in ("Pu", "Pz"):
        # These are upper tail, so we multiply by -1 to make lower tail
        stat = -1 * stat
    tau_max = PVAL_TAU_MAX[key]
    tau_min = PVAL_TAU_MIN[key]
    tau_star = PVAL_TAU_STAR[key]
    if stat > tau_max:
        return 1.0
    elif stat < tau_min:
        return 0.0
    if stat > tau_star:
        params = np.array(PVAL_LARGE_P[key])
    else:
        params = np.array(PVAL_SMALL_P[key])
    order = params.shape[0]
    x = stat ** np.arange(order)
    return stats.norm().cdf(params @ x)
