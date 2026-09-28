"""
Censored/latent-variance GARCH(p, q).

Implements the variance-recursion correction described in Morgan & Trevor
(1999) and used in subsequent studies of censored return series (e.g. prices
subject to an exchange circuit-breaker or price-limit band): when the return
at time t is known to be censored (the true, latent shock would have exceeded
a known threshold in absolute value, but only the threshold itself is
observed), using the raw observed (truncated) squared residual in the GARCH
recursion understates the true conditional variance, because it discards the
information that the *true* shock was larger than what was recorded.

This class corrects the recursion by replacing the squared residual at a
censored observation with its conditional expectation given censoring, under
the model's own (currently Gaussian) innovation distribution:

.. math::

    E[\\varepsilon_t^2 \\mid |\\varepsilon_t| \\ge c_t, \\mathcal{F}_{t-1}]
        = \\sigma_t^2 \\left(1 + a_t \\cdot \\frac{\\phi(a_t)}{1 - \\Phi(a_t)}\\right),
    \\quad a_t = c_t / \\sigma_t

where :math:`c_t` is the (known) censoring threshold in effect at time t,
:math:`\\phi` and :math:`\\Phi` are the standard normal pdf/cdf, and
:math:`\\sigma_t^2` is the model's own conditional variance at t (available
within the recursion since it is computed sequentially forward). This is the
standard truncated-normal second-moment identity applied to a two-sided
(symmetric) censoring rule.

Two importance caveats, stated in Notes on the class itself:

* The correction uses the *model's own* :math:`\\sigma_t` for uncensored
  and censored steps alike; it therefore self-consistently propagates the
  extra variance from censored periods into subsequent, uncensored ones
  through the ordinary GARCH recursion.
* The correction is derived for Gaussian innovations. The recursion is
  still usable, as an approximation, with a non-Gaussian ``Distribution``
  (e.g. Student's t) selected for the *overall* model likelihood, but the
  correction factor itself is not re-derived for that case.

See the accompanying paper referenced in the class docstring for the
underlying empirical motivation and validation.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
from scipy.stats import norm

from arch._typing import (
    ArrayLike1D,
    Float64Array,
    Float64Array1D,
    Float64Array2D,
    ForecastingMethod,
    RNGType,
)
from arch.univariate.recursions_python import bounds_check_python as bounds_check
from arch.univariate.volatility import VarianceForecast, VolatilityProcess
from arch.utility.array import AbstractDocStringInheritor, ensure1d, to_array_1d

__all__ = ["CensoredGARCH"]


def censored_garch_recursion(
    parameters: Float64Array1D,
    resids: Float64Array1D,
    censored: Float64Array1D,
    threshold: Float64Array1D,
    sigma2: Float64Array1D,
    p: int,
    q: int,
    nobs: int,
    backcast: float,
    var_bounds: Float64Array2D,
) -> Float64Array1D:
    """
    Pure-Python reference recursion for :class:`CensoredGARCH`.

    Not currently accelerated (no Cython/Numba backend) -- see the class
    docstring's "Notes" section.

    Parameters
    ----------
    parameters : ndarray
        [omega, alpha_1..alpha_p, beta_1..beta_q]
    resids : ndarray
        Mean-equation residuals. For a censored observation, this is the
        *observed* (truncated) residual, i.e. it equals +/- the threshold
        in effect at that time, not the (unobserved) true shock.
    censored : ndarray
        Same length as resids. Nonzero (truthy) where the observation at
        that index is censored.
    threshold : ndarray
        Same length as resids. The censoring threshold |c_t| in effect at
        each time t (only used where ``censored`` is truthy).
    sigma2 : ndarray
        Output array for the conditional variance.
    p, q : int
        GARCH orders (no asymmetric/leverage term in this first version).
    nobs : int
        Number of observations.
    backcast : float
        Initial value used before any real observations are available.
    var_bounds : ndarray
        Per-observation [lower, upper] bounds enforced on sigma2.

    Returns
    -------
    sigma2 : ndarray
        The conditional variance, written in place and also returned.
    """
    eff2 = np.empty(nobs, dtype=float)

    for t in range(nobs):
        loc = 0
        sigma2[t] = parameters[loc]
        loc += 1
        for j in range(p):
            idx = t - 1 - j
            if idx < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * eff2[idx]
            loc += 1
        for j in range(q):
            idx = t - 1 - j
            if idx < 0:
                sigma2[t] += parameters[loc] * backcast
            else:
                sigma2[t] += parameters[loc] * sigma2[idx]
            loc += 1

        sigma2[t] = bounds_check(sigma2[t], var_bounds[t])

        if not censored[t]:
            eff2[t] = resids[t] * resids[t]
        else:
            sigma_t = np.sqrt(sigma2[t])
            a = threshold[t] / sigma_t if sigma_t > 0 else 0.0
            if a <= 0:
                corr = 1.0
            else:
                one_minus_phi = norm.sf(a)
                if one_minus_phi < 1e-12:
                    # Far into the tail: 1 - Phi(a) underflows before a does.
                    # Asymptotically phi(a)/(1-Phi(a)) ~ a, so the correction
                    # factor 1 + a*hazard(a) -> 1 + a^2.
                    corr = 1.0 + a * a
                else:
                    corr = 1.0 + a * norm.pdf(a) / one_minus_phi
            eff2[t] = corr * sigma2[t]

    return sigma2


class CensoredGARCH(VolatilityProcess, metaclass=AbstractDocStringInheritor):
    r"""
    Censored/latent-variance GARCH(p, q) for return series subject to a
    known, time-varying censoring rule (e.g. an exchange price-limit band).

    Parameters
    ----------
    censored : {ndarray, Series}
        Boolean (or 0/1) array, same length as the data used to fit the
        model, indicating which observations are censored (the recorded
        return equals the threshold in effect that period, rather than the
        true underlying shock).
    threshold : {ndarray, Series, float}
        The censoring threshold |c_t| in effect at each time t, in the same
        units as the model residuals. May be passed as a scalar if the
        threshold is constant over the sample.
    p : int, optional
        Order of the symmetric innovation. Default is 1.
    q : int, optional
        Order of the lagged conditional variance. Default is 1.

    Examples
    --------
    >>> from arch.univariate import CensoredGARCH
    >>> cens = (returns.abs() >= 0.0995)  # e.g. a +/-10% price-limit band
    >>> vol = CensoredGARCH(censored=cens, threshold=0.0995)

    Notes
    -----
    In this class of processes, the variance dynamics are

    .. math::

        \sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i \tilde\varepsilon_{t-i}^2
                      + \sum_{k=1}^q \beta_k \sigma_{t-k}^2

    where :math:`\tilde\varepsilon_s^2 = \varepsilon_s^2` when observation
    :math:`s` is not censored, and

    .. math::

        \tilde\varepsilon_s^2 = \sigma_s^2
            \left(1 + a_s \frac{\phi(a_s)}{1-\Phi(a_s)}\right), \quad
            a_s = c_s / \sigma_s

    when it is (the conditional expectation of the squared shock given that
    it exceeded the threshold, under Gaussian innovations). This uses the
    truncated-normal second-moment identity
    :math:`\int_a^\infty x^2\phi(x)\,dx = a\phi(a) + (1-\Phi(a))`.

    Only symmetric (no leverage/asymmetric term) GARCH(p, q) is supported in
    this first version; ``power`` is fixed at 2.0.  No accelerated
    (Cython/Numba) recursion is provided yet -- see
    :func:`censored_garch_recursion`.

    Analytic multi-step forecasting and simulation-based forecasting are not
    yet implemented (only one-step-ahead analytic forecasts, and forward
    simulation assuming no future censoring, are supported).
    """

    def __init__(
        self,
        censored: ArrayLike1D,
        threshold: ArrayLike1D | float,
        p: int = 1,
        q: int = 1,
    ) -> None:
        super().__init__()
        if p < 0 or q < 0 or (p == 0 and q == 0):
            raise ValueError("One of p or q must be strictly positive")
        self.p: int = int(p)
        self.q: int = int(q)
        self._num_params = 1 + self.p + self.q
        self._name = f"Censored-GARCH({self.p}, {self.q})"
        self._updatable = False

        censored_arr = ensure1d(censored, "censored", True)
        self._censored = np.asarray(censored_arr, dtype=bool)

        if np.isscalar(threshold):
            self._threshold = np.full(
                self._censored.shape[0], float(threshold), dtype=float
            )
        else:
            self._threshold = to_array_1d(
                ensure1d(threshold, "threshold", True)
            ).astype(float)
            if self._threshold.shape[0] != self._censored.shape[0]:
                raise ValueError(
                    "censored and threshold must be the same length when "
                    "threshold is not a scalar"
                )
        if np.any(self._threshold[self._censored] <= 0):
            raise ValueError(
                "threshold must be strictly positive wherever censored is True"
            )

    def bounds(self, resids: ArrayLike1D) -> list[tuple[float, float]]:
        v = float(np.mean(np.asarray(resids) ** 2))
        bounds = [(1e-8 * v, 10.0 * v)]
        bounds.extend([(0.0, 1.0)] * self.p)
        bounds.extend([(0.0, 1.0)] * self.q)
        return bounds

    def constraints(self) -> tuple[Float64Array, Float64Array]:
        k = self.p + self.q
        a = np.zeros((k + 2, k + 1))
        for i in range(k + 1):
            a[i, i] = 1.0
        a[k + 1, 1:] = -1.0
        b = np.zeros(k + 2)
        b[k + 1] = -1.0
        return a, b

    def _slice(self, arr: Float64Array1D) -> Float64Array1D:
        return arr[self._start : self._stop]

    def compute_variance(
        self,
        parameters: Float64Array1D,
        resids: ArrayLike1D,
        sigma2: Float64Array1D,
        backcast: float | Float64Array1D,
        var_bounds: Float64Array2D,
    ) -> Float64Array1D:
        _resids = to_array_1d(resids)
        nobs = _resids.shape[0]
        censored = self._slice(self._censored)
        threshold = self._slice(self._threshold)
        if censored.shape[0] == nobs - 1:
            # `_one_step_forecast` (the base class helper used by
            # `_analytic_forecast`) appends one extra placeholder residual
            # (0.0) past the end of the fitted sample to obtain a one-step
            # ahead variance forecast. That placeholder observation isn't
            # part of the real censored/threshold series, and it is never
            # itself fed back into the recursion (it is always the last
            # point), so it is safe to pad with an "uncensored" placeholder.
            censored = np.concatenate([censored, np.array([False])])
            threshold = np.concatenate([threshold, np.array([0.0])])
        elif censored.shape[0] != nobs:
            raise ValueError(
                "The length of the censored/threshold arrays passed at "
                "construction does not match the data used to fit the "
                "model. Re-check `start`/`stop` slicing."
            )
        assert isinstance(backcast, float)
        censored_garch_recursion(
            parameters,
            _resids,
            censored,
            threshold,
            sigma2,
            self.p,
            self.q,
            nobs,
            backcast,
            var_bounds,
        )
        return sigma2

    def starting_values(self, resids: ArrayLike1D) -> Float64Array1D:
        p, q = self.p, self.q
        target = float(np.mean(np.asarray(resids) ** 2))
        var_bounds = self.variance_bounds(resids)
        backcast = self.backcast(resids)

        alphas = [0.05, 0.1, 0.2]
        persistences = [0.7, 0.9, 0.98]
        svs: list[Float64Array1D] = []
        llfs = []
        for alpha_total, persistence in [(a, b) for a in alphas for b in persistences]:
            sv = np.zeros(1 + p + q, dtype=float)
            sv[0] = (1.0 - persistence) * target
            if p > 0:
                sv[1 : 1 + p] = alpha_total / p
            if q > 0:
                sv[1 + p : 1 + p + q] = (persistence - alpha_total) / q
                sv[1 + p : 1 + p + q] = np.clip(sv[1 + p : 1 + p + q], 1e-4, None)
            svs.append(cast("Float64Array1D", sv))
            llfs.append(self._gaussian_loglikelihood(sv, resids, backcast, var_bounds))
        loc = int(np.argmax(llfs))
        return svs[loc]

    def parameter_names(self) -> list[str]:
        names = ["omega"]
        names.extend([f"alpha[{i + 1}]" for i in range(self.p)])
        names.extend([f"beta[{i + 1}]" for i in range(self.q)])
        return names

    def simulate(
        self,
        parameters: Sequence[int | float] | ArrayLike1D,
        nobs: int,
        rng: RNGType,
        burn: int = 500,
        initial_value: float | Float64Array | None = None,
    ) -> tuple[Float64Array, Float64Array]:
        """
        Simulate from the model, assuming NO censoring in the simulated
        path (future censoring status is not known ex ante -- this
        simulates the *unconditional* process the model implies, which is
        only exactly correct when the true return process is never
        censored; it is an approximation whenever a future censoring rule
        would actually bind). See class Notes.
        """
        parameters = ensure1d(parameters, "parameters", False)
        p, q = self.p, self.q
        errors = rng(nobs + burn)

        if initial_value is None:
            persistence = float(np.sum(parameters[1:]))
            if (1.0 - persistence) > 0:
                initial_value = parameters[0] / (1.0 - persistence)
            else:
                initial_value = parameters[0]

        sigma2 = np.zeros(nobs + burn)
        data = np.zeros(nobs + burn)
        max_lag = max(p, q, 1)
        sigma2[:max_lag] = initial_value
        data[:max_lag] = np.sqrt(sigma2[:max_lag]) * errors[:max_lag]

        for t in range(max_lag, nobs + burn):
            loc = 0
            sigma2[t] = parameters[loc]
            loc += 1
            for j in range(p):
                sigma2[t] += parameters[loc] * data[t - 1 - j] ** 2
                loc += 1
            for j in range(q):
                sigma2[t] += parameters[loc] * sigma2[t - 1 - j]
                loc += 1
            data[t] = errors[t] * np.sqrt(sigma2[t])

        return data[burn:], sigma2[burn:]

    def _check_forecasting_method(
        self, method: ForecastingMethod, horizon: int
    ) -> None:
        if horizon > 1:
            raise NotImplementedError(
                "Multi-step forecasts are not yet implemented for "
                "CensoredGARCH; only horizon=1 is supported."
            )
        if method != "analytic":
            raise NotImplementedError(
                "Only method='analytic' is currently supported for "
                "CensoredGARCH."
            )

    def _analytic_forecast(
        self,
        parameters: Float64Array1D,
        resids: ArrayLike1D,
        backcast: float | Float64Array1D,
        var_bounds: Float64Array2D,
        start: int,
        horizon: int,
    ) -> VarianceForecast:
        sigma2, forecasts = self._one_step_forecast(
            parameters, to_array_1d(resids), backcast, var_bounds, horizon, start
        )
        return VarianceForecast(forecasts)

    def _simulation_forecast(
        self,
        parameters: Float64Array1D,
        resids: ArrayLike1D,
        backcast: float | Float64Array1D,
        var_bounds: Float64Array2D,
        start: int,
        horizon: int,
        simulations: int,
        rng: RNGType,
    ) -> VarianceForecast:
        raise NotImplementedError(
            "Simulation-based forecasts are not yet implemented for "
            "CensoredGARCH."
        )
