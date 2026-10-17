from collections.abc import Hashable, Sequence
import copy
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len, rfft

from arch._typing import (
    ArrayLike,
    ArrayLike2D,
    BoolArray,
    Float64Array,
    Int64Array1D,
    IntArray,
    Literal,
)
from arch.bootstrap.base import (
    CircularBlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
    _get_random_integers,
)
from arch.utility._validation import (
    check_array,
    check_bool,
    check_float,
    check_int,
    check_option,
    check_seed,
)
from arch.utility.array import DocStringInheritor, as_float64, ensure2d

__all__ = ["MCS", "SPA", "RealityCheck", "StepM"]

_PVALUE_TYPES = ("lower", "consistent", "upper")

# Lower case names of the MCS methods, which map to the canonical names
_MCS_METHODS = {"r": "R", "max": "max"}

# Lower case names of the bootstraps, using underscores instead of spaces
_BOOTSTRAPS: dict[str, type[CircularBlockBootstrap]] = {
    "circular": CircularBlockBootstrap,
    "cbb": CircularBlockBootstrap,
    "stationary": StationaryBootstrap,
    "sb": StationaryBootstrap,
    "moving_block": MovingBlockBootstrap,
    "mbb": MovingBlockBootstrap,
}


def _bootstrap_class(bootstrap: str) -> type[CircularBlockBootstrap]:
    """Find the bootstrap class from its name, ignoring case and spaces"""
    name = check_option(
        "bootstrap",
        bootstrap,
        tuple(_BOOTSTRAPS),
        display=("stationary", "sb", "circular", "cbb", "moving block", "mbb"),
        normalize=lambda value: value.replace(" ", "_"),
    )
    return _BOOTSTRAPS[name]


# The smallest number of observations. The SPA needs log(log(t)) > 0. The MCS of a
# single observation runs and warns that the MCS is not valid for this problem.
_MIN_OBS_MCS = 1
_MIN_OBS_SPA = 3


def _check_data(name: str, value: object, **kwargs: Any) -> np.ndarray:
    """
    Check the shape of losses, allowing NaN and inf that are rejected later.

    Data that cannot be converted to numbers keep the message of the conversion,
    which says that float64 is required.
    """
    try:
        return check_array(name, value, allow_nan=True, allow_inf=True, **kwargs)
    except TypeError:
        as_float64(value, name)
        raise


def _require_columns(name: str, arr: np.ndarray, minimum: int, value: object) -> None:
    """Check the number of columns of 2-d data that has been checked to be 2-d"""
    if arr.shape[1] < minimum:
        plural = "" if minimum == 1 else "s"
        raise ValueError(
            f"{name} must be a 2-d array with at least {minimum} column{plural}; "
            f"got shape {arr.shape} ({type(value).__name__})"
        )


def _check_resampling(
    reps: object, block_size: object, seed: object, t: int
) -> tuple[int, int | None, int | np.random.Generator | np.random.RandomState | None]:
    """Check the arguments that control the bootstrap and that all procedures share"""
    reps = check_int("reps", reps, lower=1)
    block_size = check_int("block_size", block_size, lower=1, upper=t, allow_none=True)
    return reps, block_size, check_seed("seed", seed)


def _info_to_str(
    model: str, info: dict[str, str], is_repr: bool = False, is_html: bool = False
) -> str:
    if is_html:
        model = "<strong>" + model + "</strong>"
    _str = model + "("
    for k, v in info.items():
        if k.lower() != "id" or is_repr:
            _k = k
            if is_html:
                _k = "<strong>" + _k + "</strong>"
            _str += _k + ": " + v + ", "
    return _str[:-2] + ")"


_NON_FINITE_STATISTIC = (
    "The MCS test statistic is not finite so no model can be eliminated. This "
    "occurs when the losses are not finite."
)


class _KernelVariance:
    """
    Long-run variance of the mean using the kernel implied by a bootstrap

    Parameters
    ----------
    bootstrap : CircularBlockBootstrap
        The bootstrap that determines the kernel.
    t : int
        Number of observations in the data that will be passed when the
        instance is called.

    Notes
    -----
    Computes only the variance of each column, not the covariance, using a
    single FFT so that the cost is O(t log t) rather than O(t * bandwidth)
    for each column.
    """

    def __init__(self, bootstrap: CircularBlockBootstrap, t: int) -> None:
        weights = self._implied_kernel(bootstrap, t)
        kernel: Float64Array
        if isinstance(bootstrap, MovingBlockBootstrap):
            # Sample autocovariances without wrap-around. Padding avoids
            # aliasing of the lags in the support of the kernel.
            self._n = next_fast_len(t + int(np.flatnonzero(weights)[-1]), real=True)
            kernel = np.zeros(self._n)
            kernel[:t] = weights
            kernel[self._n - t + 1 :] += weights[:0:-1]
        else:
            # Weights are symmetric, w[i] = w[t - i], so circular
            # autocovariances give the same sum without padding.
            self._n = t
            kernel = weights
        spectrum = rfft(kernel).real
        multiplicity = np.full(spectrum.shape[0], 2.0)
        multiplicity[0] = 1.0
        if self._n % 2 == 0:
            multiplicity[-1] = 1.0
        self._spectral_weights = multiplicity * spectrum / (t * self._n)

    @staticmethod
    def _implied_kernel(bootstrap: CircularBlockBootstrap, t: int) -> Float64Array:
        """
        Autocovariance weights of the variance of the mean implied by a bootstrap

        Parameters
        ----------
        bootstrap : CircularBlockBootstrap
            The bootstrap. Stationary, circular and moving block bootstraps are
            recognized, and any other subclass of CircularBlockBootstrap is
            treated as a circular block bootstrap.
        t : int
            Number of observations.

        Returns
        -------
        ndarray
            Array with t elements containing the weights, w, so that the variance
            of the sample mean times t is estimated using
            ``R(0) + 2 * sum(w[i] * R(i))`` where ``R`` is the sample
            autocovariance.

        Notes
        -----
        The weights for the stationary bootstrap and the circular block bootstrap
        are the weights that produce the exact bootstrap variance of the sample
        mean [1]_. The circular block bootstrap weights are the Bartlett kernel
        with a bandwidth of the block size, adjusted for the wrap-around of the
        data and for the truncation of the final block when the block size does
        not divide t. The weights for the moving block bootstrap are the Bartlett
        kernel without the wrap-around adjustment, which is asymptotically
        equivalent to the variance of the mean from the moving block bootstrap.

        References
        ----------
        .. [1] Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
           Journal of the American Statistical Association, 89(428), 1303-1313.
        """
        lags = np.arange(t, dtype=float)
        block_size = bootstrap.block_size
        if isinstance(bootstrap, StationaryBootstrap):
            rho = 1.0 - 1.0 / block_size
            return (1.0 - lags / t) * rho**lags + (lags / t) * rho ** (t - lags)

        num_blocks, remainder = divmod(t, block_size)

        def bartlett(lag: Float64Array) -> Float64Array:
            # Blocks of block_size and a final block truncated to remainder
            full = num_blocks * np.maximum(block_size - lag, 0.0)
            return (full + np.maximum(remainder - lag, 0.0)) / t

        weights = bartlett(lags)
        if not isinstance(bootstrap, MovingBlockBootstrap):
            # Blocks wrap around the end of the sample
            weights[1:] += bartlett(t - lags[1:])
        return weights

    def __call__(self, x: Float64Array) -> Float64Array:
        """
        Estimate the long-run variance of the mean of each column

        Parameters
        ----------
        x : ndarray
            Array with t rows. The mean of each column is removed.

        Returns
        -------
        ndarray
            The estimated variance of the sample mean times t of each column.
        """
        transform = rfft(x - x.mean(axis=0), n=self._n, axis=0)
        periodogram = transform.real**2 + transform.imag**2
        # Rounding can produce tiny negative values when variance is 0
        return np.maximum(self._spectral_weights @ periodogram, 0.0)


class MultipleComparison:
    """
    Abstract class for inheritance.
    """

    def __init__(self) -> None:
        self._model = ""
        self._info: dict[str, str] = {}
        self._bootstrap: CircularBlockBootstrap = CircularBlockBootstrap(
            10, np.ones(100)
        )

    @property
    def bootstrap(self) -> CircularBlockBootstrap:
        """
        The bootstrap used in the comparison.

        Returns
        -------
        CircularBlockBootstrap
            The stationary, circular block or moving block bootstrap. The
            bootstrap cannot be replaced since quantities that depend on it,
            such as the kernel used to studentize, are computed when the
            instance is created.
        """
        return self._bootstrap

    def __str__(self) -> str:
        return _info_to_str(self._model, self._info, False)

    def __repr__(self) -> str:
        return _info_to_str(self._model, self._info, True)

    def _repr_html_(self) -> str:
        return _info_to_str(self._model, self._info, True, True)

    def reset(self) -> None:
        """
        Reset the bootstrap to its initial state.
        """
        self.bootstrap.reset()


class MCS(MultipleComparison):
    """
    Model Confidence Set (MCS) of Hansen, Lunde and Nason.

    Parameters
    ----------
    losses : ndarray or DataFrame
        T by k array containing losses from a set of models. There must be at
        least one row and two columns, and the losses must be finite.
    size : float
        Value in (0,1) to use as the test size when implementing the
        MCS. There is no default value.
    reps : int, optional
        Number of bootstrap replications to use. Must be at least 1.
    block_size : int, optional
        Length of window to use in the bootstrap, between 1 and T.  If not
        provided, sqrt(T) is used.  In general, this should be provided and
        chosen to be appropriate for the data.
    method : {"R", "max"}, optional
        MCS test and elimination implementation method, either "max" or "R".
    bootstrap : {"stationary", "sb", "circular", "cbb", "moving block", "mbb"}, optional
        Bootstrap to use.  Options are

        * "stationary" or "sb": Stationary bootstrap
        * "circular" or "cbb": Circular block bootstrap
        * "moving block" or "mbb": Moving block bootstrap
    seed : int, Generator or RandomState, optional
        Seed value to use when creating the bootstrap used in the comparison.
        If an integer or None, the NumPy default_rng is used with the seed
        value.  If a Generator or a RandomState, the argument is used.

    Raises
    ------
    TypeError
        If reps or block_size is not an integer, size is not a real number,
        seed is not an integer, a Generator, a RandomState or None, an option
        is not a string, or the losses cannot be converted to float64.
    ValueError
        If the losses are not 2-d with at least one row and two columns, or
        contain NaN or inf, size is not in (0,1), reps is less than 1,
        block_size is not between 1 and T, seed is negative, or an option is
        not one of the allowed values.

    Notes
    -----
    See [1]_ for details.

    References
    ----------
    .. [1] Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
       Econometrica, 79(2), 453-497.
    """

    def __init__(
        self,
        losses: ArrayLike2D,
        size: float,
        reps: int = 1000,
        block_size: int | None = None,
        method: Literal["R", "max"] = "R",
        bootstrap: Literal[
            "stationary", "sb", "circular", "cbb", "moving block", "mbb"
        ] = "stationary",
        *,
        seed: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> None:
        super().__init__()
        # Non-finite losses are rejected below with a message about the losses
        _check_data("losses", losses, ndim=2, min_length=_MIN_OBS_MCS)
        self.losses = ensure2d(losses, "losses")
        self._losses_arr = np.asarray(self.losses, dtype=float)
        if not np.all(np.isfinite(self._losses_arr)):
            raise ValueError(
                "losses must be finite. The losses contain NaN or infinite values."
            )
        _require_columns("losses", self._losses_arr, 2, losses)
        self.size: float = check_float(
            "size", size, lower=0.0, upper=1.0, inclusive="neither"
        )
        reps, block_arg, seed = _check_resampling(
            reps, block_size, seed, self.losses.shape[0]
        )
        self.reps: int = reps
        if block_arg is None:
            self.block_size = int(np.sqrt(self.losses.shape[0]))
        else:
            self.block_size = block_arg

        self.t: int = self.losses.shape[0]
        self.k: int = self.losses.shape[1]
        self.method: Literal["R", "max"] = cast(
            "Literal['R', 'max']",
            check_option("method", method, _MCS_METHODS, display=("R", "max")),
        )
        # Bootstrap indices since the same bootstrap should be used in the
        # repeated steps
        indices = np.arange(self.t)
        bootstrap_cls = _bootstrap_class(bootstrap)
        bootstrap_inst = bootstrap_cls(self.block_size, indices, seed=seed)
        self._seed = seed
        self._bootstrap = bootstrap_inst
        # Optional index retention, for testing
        self._store_index: bool = False
        self._bootstrap_indices: list[IntArray] = []
        self._model = "MCS"
        self._info = {
            "size": f"{self.size:0.2f}",
            "bootstrap": str(bootstrap_inst),
            "ID": hex(id(self)),
        }
        self._results_computed = False

    @property
    def _testing(self) -> bool:
        """Flag indicating to store bootstrap indices for testing purposes"""
        return self._store_index

    @_testing.setter
    def _testing(self, value: bool) -> None:
        self._store_index = bool(value)

    def _has_been_computed(self) -> None:
        if not self._results_computed:
            raise RuntimeError("Must call compute before accessing results")

    def _format_pvalues(self, eliminated: Sequence[tuple[int, float]]) -> pd.DataFrame:
        columns = ["Model index", "Pvalue"]
        mcs = pd.DataFrame(eliminated, columns=columns)
        max_pval = cast("float", mcs.iloc[0, 1])
        for i in range(1, mcs.shape[0]):
            max_pval = np.max([max_pval, cast("float", mcs.iloc[i, 1])])
            mcs.iloc[i, 1] = max_pval
        model_index = mcs.pop("Model index")
        if isinstance(self.losses, pd.DataFrame):
            # Workaround for old pandas/numpy combination
            # Preferred expression :
            model_index = pd.Series(self.losses.columns[model_index])
            # model_index = self.losses.iloc[:, model_index.to_numpy()].columns
            # model_index = pd.Series(model_index)
            model_index.name = "Model name"
        mcs.index = pd.Index(model_index)
        return mcs

    def compute(self) -> None:
        """
        Compute the set of models in the confidence set.
        """
        # The method is a public attribute that can be set after construction
        method = check_option("method", self.method, _MCS_METHODS, display=("R", "max"))
        if method == "R":
            self._compute_r()
        else:
            self._compute_max()
        self._results_computed = True

    def _compute_r(self) -> None:
        """
        Computes the model confidence set using the R method
        """
        # R method
        # 1. Compute pairwise difference (k,k)
        losses = self._losses_arr
        mean_losses = losses.mean(axis=0)[:, None]
        loss_diffs = mean_losses - mean_losses.T
        # Compute pairwise variance using bootstrap (k,k)
        # In each bootstrap, save the average difference of each pair (b,k,k)
        bootstrapped_mean_losses = np.zeros((self.reps, self.k, self.k))
        bs = self.bootstrap
        for j in range(self.reps):
            # The bootstrapped data are the indices themselves, so the indices
            # are drawn directly rather than resampled from np.arange(t)
            bs_index = bs.update_indices()
            if self._store_index:
                # For testing
                self._bootstrap_indices.append(np.asarray(bs_index, dtype=int))
            # The mean of the resampled rows weights each row by the number of
            # times it is drawn, which avoids copying the resampled losses
            counts = np.bincount(bs_index, minlength=self.t)
            mean_losses_star = (counts @ losses / self.t)[:, None]
            bootstrapped_mean_losses[j] = mean_losses_star - mean_losses_star.T
        # Recenter
        bootstrapped_mean_losses -= loss_diffs
        variances = (bootstrapped_mean_losses**2).mean(axis=0)
        variances += np.eye(self.k)  # Prevent division by 0
        self._variances = variances
        # Standardize everything
        std_loss_diffs = loss_diffs / np.sqrt(variances)
        std_bs_mean_losses = bootstrapped_mean_losses / np.sqrt(variances)
        # 3. Using the models still in the set, compute the max (b,1)
        # Initialize the set
        included = np.ones(self.k, dtype=np.bool_)
        # Loop until there is only 1 model left
        eliminated = []
        while included.sum() > 1:
            indices = np.argwhere(included)
            included_loss_diffs = std_loss_diffs[indices, indices.T]
            test_stat = np.max(included_loss_diffs)
            included_bs_losses = std_bs_mean_losses[:, indices, indices.T]
            simulated_test_stat = np.max(np.max(included_bs_losses, 2), 1)
            pval = float((test_stat < simulated_test_stat).mean())
            loc = np.argwhere(included_loss_diffs == test_stat)
            if loc.shape[0] == 0:
                # Only possible if the statistic is not finite
                raise RuntimeError(_NON_FINITE_STATISTIC)
            # Loc has indices i,j, -- i  is the elimination
            # Diffs are L(i) - L(j), so large value in [i,j] indicates
            # i is worse than j
            # Elimination is for
            # The first is used if several models are tied
            i = loc[0, 0]
            eliminated.append((int(indices.flat[i]), pval))
            included[indices.flat[i]] = False
        # Add pval of 1 for model remaining
        remaining_indices = np.argwhere(included).flatten()
        eliminated.extend([(int(ind), 1.0) for ind in remaining_indices])
        self._pvalues = self._format_pvalues(eliminated)

    def _compute_max(self) -> None:
        """
        Computes the model confidence set using the R method
        """
        # max method
        losses = self._losses_arr
        # 1. compute loss "errors"
        loss_errors = losses - losses.mean(axis=0)
        # Generate bootstrap samples
        bs_avg_loss_errors = np.zeros((self.reps, self.k))
        for i in range(self.reps):
            bs_index = self.bootstrap.update_indices()
            if self._store_index:
                # For testing
                self._bootstrap_indices.append(np.asarray(bs_index, dtype=int))
            counts = np.bincount(bs_index, minlength=self.t)
            avg_bs_errors = counts @ loss_errors / self.t
            avg_bs_errors -= avg_bs_errors.mean()
            bs_avg_loss_errors[i] = avg_bs_errors
            # Initialize the set
        included = np.ones(self.k, dtype=np.bool_)
        # Loop until there is only 1 model left
        eliminated = []
        while included.sum() > 1:
            indices = np.argwhere(included)
            incl_losses = losses[:, included]
            incl_bs_avg_loss_err = bs_avg_loss_errors[:, included]
            incl_bs_grand_loss = incl_bs_avg_loss_err.mean(1)
            # Reshape for broadcast
            incl_bs_avg_loss_err -= incl_bs_grand_loss[:, None]
            std_devs = np.sqrt((incl_bs_avg_loss_err**2).mean(axis=0))
            if np.any(std_devs <= 0):
                warnings.warn(
                    "During computation of a step of the MCS the estimated standard "
                    "deviation of at least one loss difference was 0.  This "
                    "indicates that the MCS is not valid for this problem. This can "
                    "occur if the number of losses is too small, or if there are "
                    "repeated (identical) losses in the set under consideration.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                std_devs[std_devs <= 0] = np.finfo(float).eps
            simulated_test_stat = incl_bs_avg_loss_err / std_devs
            simulated_test_stat = np.max(simulated_test_stat, 1)
            loss_diffs = incl_losses.mean(axis=0)
            loss_diffs -= loss_diffs.mean()
            std_loss_diffs = loss_diffs / std_devs
            test_stat = np.max(std_loss_diffs)
            pval = float((test_stat < simulated_test_stat).mean())
            locs = np.argwhere(std_loss_diffs == test_stat)
            if locs.shape[0] == 0:
                # Only possible if the statistic is not finite. Without a model
                # to eliminate the loop would never end.
                raise RuntimeError(_NON_FINITE_STATISTIC)
            eliminated.extend(
                [
                    (int(idx_val), pval)
                    for idx_val in indices.flat[np.atleast_1d(locs.squeeze())]
                ]
            )
            included[indices.flat[locs]] = False

        remaining_indices = np.argwhere(included).flatten()
        eliminated.extend([(int(ind), 1.0) for ind in remaining_indices])
        self._pvalues = self._format_pvalues(eliminated)

    @property
    def included(self) -> list[Hashable]:
        """
        List of model indices that are included in the MCS.

        Returns
        -------
        included : list
            List of column indices or names of the included models, which are
            the models with a p-value that is at least size.
        """
        self._has_been_computed()
        incl_loc = self._pvalues.Pvalue >= self.size
        included = list(self._pvalues.index[incl_loc])
        included.sort()
        return included

    @property
    def excluded(self) -> list[Hashable]:
        """
        List of model indices that are excluded from the MCS.

        Returns
        -------
        excluded : list
            List of column indices or names of the excluded models, which are
            the models with a p-value that is less than size.
        """
        self._has_been_computed()
        excl_loc = self._pvalues.Pvalue < self.size
        excluded = list(self._pvalues.index[excl_loc])
        excluded.sort()
        return excluded

    @property
    def pvalues(self) -> pd.DataFrame:
        """
        Model p-values for inclusion in the MCS.

        Returns
        -------
        pvalues : DataFrame
            DataFrame where the index is the model index (column or name)
            and the column Pvalue contains the MCS p-value of the model. The
            model is in the MCS of size alpha, and so in ``included``, if its
            p-value is at least alpha. The models are in the order that
            they were eliminated and the p-value of the remaining model is 1.
        """
        self._has_been_computed()
        return self._pvalues


class StepM(MultipleComparison):
    """
    StepM multiple comparison procedure of Romano and Wolf.

    Parameters
    ----------
    benchmark : ndarray or Series
        T element array of benchmark model *losses*.  There must be at least
        three observations, and the losses must be finite.
    models : ndarray or DataFrame
        T by k element array of alternative model *losses*, with the same T as
        benchmark and k at least 1.
    size : float, optional
        Value in (0,1) to use as the test size when implementing the
        comparison.
    block_size : int, optional
        Length of window to use in the bootstrap, between 1 and T.  If not
        provided, sqrt(T) is used.  In general, this should be provided and
        chosen to be appropriate for the data.
    reps : int, optional
        Number of bootstrap replications to use. Must be at least 1.
    bootstrap : {"stationary", "sb", "circular", "cbb", "moving block", "mbb"}, optional
        Bootstrap to use.  Options are

        * "stationary" or "sb": Stationary bootstrap
        * "circular" or "cbb": Circular block bootstrap
        * "moving block" or "mbb": Moving block bootstrap
    studentize : bool, optional
        Flag indicating to studentize loss differentials.
        Studentization is also applied inside each bootstrap replication,
        see :class:`~arch.bootstrap.SPA`. If False, the loss differentials
        are not studentized.
    nested : bool, optional
        Flag indicating to use a nested bootstrap to compute the variances
        used to studentize, instead of the kernel estimator of the long-run
        variance that matches the bootstrap. The variances are computed in
        the original data and in each bootstrap replication, so the
        procedure uses reps squared resamples and can be very slow. Can only
        be True when studentize is True.
    seed : int, Generator or RandomState, optional
        Seed value to use when creating the bootstrap used in the comparison.
        If an integer or None, the NumPy default_rng is used with the seed
        value.  If a Generator or a RandomState, the argument is used.

    Raises
    ------
    TypeError
        If reps or block_size is not an integer, size is not a real number,
        studentize or nested is not a bool, seed is not an integer, a
        Generator, a RandomState or None, bootstrap is not a string, or the
        losses cannot be converted to float64.
    ValueError
        If benchmark is not 1-d, models is not 2-d with T rows, there are fewer
        than three observations, the losses contain NaN or inf, size is not in
        (0,1), reps is less than 1, block_size is not between 1 and T, seed is
        negative, nested is True and studentize is False, or bootstrap is not
        one of the allowed values.

    See Also
    --------
    arch.bootstrap.SPA
        Test of Superior Predictive Ability of White and Hansen.

    Notes
    -----
    The size controls the Family Wise Error Rate (FWER) since this is a
    multiple comparison procedure.  Uses SPA and the consistent selection
    procedure.

    See :class:`~arch.bootstrap.SPA` for a description of how loss
    differentials are studentized.

    See [1]_ for detail.

    References
    ----------
    .. [1] Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as
       formalized data snooping. Econometrica, 73(4), 1237-1282.
    """

    def __init__(
        self,
        benchmark: ArrayLike,
        models: ArrayLike,
        size: float = 0.05,
        block_size: int | None = None,
        reps: int = 1000,
        bootstrap: Literal[
            "stationary", "sb", "circular", "cbb", "moving block", "mbb"
        ] = "stationary",
        studentize: bool = True,
        nested: bool = False,
        *,
        seed: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> None:
        super().__init__()
        self.size: float = check_float(
            "size", size, lower=0.0, upper=1.0, inclusive="neither"
        )
        # The SPA checks the data and the arguments that it shares with StepM
        self.spa: SPA = SPA(
            benchmark,
            models,
            block_size=block_size,
            reps=reps,
            bootstrap=bootstrap,
            studentize=studentize,
            nested=nested,
            seed=seed,
        )
        self.benchmark = self.spa.benchmark
        self.models = self.spa.models
        self.block_size: int = self.spa.block_size
        self.t: int = self.models.shape[0]
        self.k: int = self.models.shape[1]
        self.reps: int = self.spa.reps
        self._superior_models: list[int] | None = None
        self._bootstrap = self.spa.bootstrap

        self._model = "StepM"
        if self.spa.studentize:
            method = "bootstrap" if self.spa.nested else "asymptotic"
        else:
            method = "none"
        self._info = {
            "FWER (size)": f"{self.size:0.2f}",
            "studentization": method,
            "bootstrap": str(self.spa.bootstrap),
            "ID": hex(id(self)),
        }

    def compute(self) -> None:
        """
        Compute the set of superior models.
        """
        # 1. Run SPA
        self.spa.compute()
        # 2. If any models superior, store indices, remove and re-run SPA
        better_models = [int(i) for i in self.spa.better_models(self.size)]
        all_better_models = better_models[:]
        # 3. Stop if nothing superior
        while better_models and (len(all_better_models) < self.k):
            # A. Use Selector to remove better models
            selector = np.ones(self.k, dtype=np.bool_)
            selector[np.array(all_better_models)] = False
            self.spa.subset(selector)
            # B. Rerun
            self.spa.compute()
            better_models = list(self.spa.better_models(self.size))
            all_better_models.extend(better_models)
        # Reset SPA
        selector = np.ones(self.k, dtype=np.bool_)
        self.spa.subset(selector)
        all_better_models.sort()
        self._superior_models = all_better_models

    @property
    def superior_models(self) -> list[int] | Sequence[Hashable]:
        """
        List of the indices or column names of the superior models.

        Returns
        -------
        list
            List of superior models.  Contains column indices if models is an
            array or contains column names if models is a DataFrame.
        """
        if self._superior_models is None:
            msg = "compute must be called before accessing superior_models"
            raise RuntimeError(msg)
        if isinstance(self.models, pd.DataFrame):
            return list(self.models.columns[self._superior_models])
        return self._superior_models


class SPA(MultipleComparison, metaclass=DocStringInheritor):
    """
    Test of Superior Predictive Ability (SPA) of White and Hansen.

    The SPA is also known as the Reality Check or Bootstrap Data Snooper.

    Parameters
    ----------
    benchmark : ndarray or Series
        T element array of benchmark model *losses*.  There must be at least
        three observations, and the losses must be finite.
    models : ndarray or DataFrame
        T by k element array of alternative model *losses*, with the same T as
        benchmark and k at least 1.
    block_size : int, optional
        Length of window to use in the bootstrap, between 1 and T.  If not
        provided, sqrt(T) is used.  In general, this should be provided and
        chosen to be appropriate for the data.
    reps : int, optional
        Number of bootstrap replications to use. Must be at least 1.
    bootstrap : {"stationary", "sb", "circular", "cbb", "moving block", "mbb"}, optional
        Bootstrap to use.  Options are

        * "stationary" or "sb": Stationary bootstrap
        * "circular" or "cbb": Circular block bootstrap
        * "moving block" or "mbb": Moving block bootstrap
    studentize : bool, optional
        Flag indicating to studentize loss differentials.
        Studentization is also applied inside each bootstrap replication,
        see Notes. If False, the loss differentials are not studentized in
        the original data or in the bootstrap replications.
    nested : bool, optional
        Flag indicating to use a nested bootstrap to compute the variances
        used to studentize, instead of the kernel estimator of the long-run
        variance that matches the bootstrap. The variances are computed in
        the original data and in each bootstrap replication, so the
        procedure uses reps squared resamples and can be very slow. Can only
        be True when studentize is True.
    seed : int, Generator or RandomState, optional
        Seed value to use when creating the bootstrap used in the comparison.
        If an integer or None, the NumPy default_rng is used with the seed
        value.  If a Generator or a RandomState, the argument is used.

    Raises
    ------
    TypeError
        If reps or block_size is not an integer, studentize or nested is not a
        bool, seed is not an integer, a Generator, a RandomState or None,
        bootstrap is not a string, or the losses cannot be converted to float64.
    ValueError
        If benchmark is not 1-d, models is not 2-d with T rows, there are fewer
        than three observations, the losses contain NaN or inf, reps is less
        than 1, block_size is not between 1 and T, seed is negative, nested is
        True and studentize is False, or bootstrap is not one of the allowed
        values.

    See Also
    --------
    arch.bootstrap.StepM
        StepM multiple comparison procedure of Romano and Wolf.

    Notes
    -----
    The three p-value correspond to different re-centering decisions.
        - Upper : Never recenter to all models are relevant to distribution
        - Consistent : Only recenter if closer than a log(log(t)) bound
        - Lower : Never recenter a model if worse than benchmark

    When ``studentize`` is True, the statistic for each model is the average
    loss differential divided by an estimate of its standard error. The
    standard error is also estimated in each bootstrap replication, using the
    bootstrap sample, and used to studentize the re-centered average of the
    bootstrap sample [3]_. Studentizing inside the bootstrap makes the
    bootstrap distribution reflect the sampling variation in the estimated
    standard errors that is present in the statistic from the original data.
    If the loss differential of a model is constant in a bootstrap sample,
    which can occur when the model differs from the benchmark in very few
    periods, the standard error from the original data is used for that
    model in that replication.

    If ``nested`` is False, the standard errors are the square root of a
    kernel estimator of the long-run variance divided by T. The kernel is
    chosen to match the bootstrap, so that the estimator is the variance of
    the sample mean that is implied by the bootstrap, using the average block
    size as the bandwidth.

        - Stationary bootstrap : The lag :math:`i` autocovariance has weight
          :math:`(1-i/T)(1-p)^i + (i/T)(1-p)^{T-i}` where :math:`p` is the
          inverse of the block size [4]_.
        - Circular block bootstrap : The Bartlett kernel applied to the
          circular autocovariances.
        - Moving block bootstrap : The Bartlett kernel.

    If ``nested`` is True, the kernel is not used. The variance is instead the
    variance of the sample mean across ``reps`` replications of a bootstrap of
    the data, which is a bootstrap of the original loss differentials in the
    original data and a bootstrap of the bootstrap sample in each replication.

    The same variances are used to select the models that are re-centered
    when computing the consistent p-value, even if ``studentize`` is False.
    No other studentization is done when ``studentize`` is False.

    If the loss differential of a model is constant in the original data, for
    example because the model is identical to the benchmark, its variance is
    0 and its test statistic is not meaningful. A ``RuntimeWarning`` is issued
    when the test is created, and such models should be removed.

    See [1]_ and [2]_ for details.

    References
    ----------
    .. [1] Hansen, P. R. (2005). A test for superior predictive ability.
       Journal of Business & Economic Statistics, 23(4), 365-380.
    .. [2] White, H. (2000). A reality check for data snooping. Econometrica,
       68(5), 1097-1126.
    .. [3] Götze, F., & Künsch, H. R. (1996). Second-order correctness of the
       blockwise bootstrap for stationary observations. The Annals of
       Statistics, 24(5), 1914-1933.
    .. [4] Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
       Journal of the American Statistical Association, 89(428), 1303-1313.
    """

    def __init__(
        self,
        benchmark: ArrayLike,
        models: ArrayLike,
        block_size: int | None = None,
        reps: int = 1000,
        bootstrap: Literal[
            "stationary", "sb", "circular", "cbb", "moving block", "mbb"
        ] = "stationary",
        studentize: bool = True,
        nested: bool = False,
        *,
        seed: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> None:
        studentize = check_bool("studentize", studentize)
        nested = check_bool("nested", nested)
        if nested and not studentize:
            raise ValueError("nested can only be True when studentize is True")
        super().__init__()
        # Non-finite values are rejected below with a message about the data
        benchmark_1d = _check_data(
            "benchmark", benchmark, ndim=1, min_length=_MIN_OBS_SPA
        )
        # Series and DataFrames keep their labels. Other data, which can be a column
        # or a row, becomes a column.
        self.benchmark = ensure2d(benchmark_1d, "benchmark")
        if isinstance(benchmark, (pd.Series, pd.DataFrame)):
            labeled = ensure2d(benchmark, "benchmark")
            if labeled.shape[1] == 1:
                self.benchmark = labeled
        _check_data("models", models, shape=(self.benchmark.shape[0], None))
        self.models = ensure2d(models, "models")
        _require_columns("models", np.asarray(self.models), 1, models)
        reps, block_arg, seed = _check_resampling(
            reps, block_size, seed, self.benchmark.shape[0]
        )
        self.reps: int = reps
        if block_arg is None:
            self.block_size = int(np.sqrt(self.benchmark.shape[0]))
        else:
            self.block_size = block_arg
        self.studentize: bool = studentize
        self.nested: bool = nested
        # Subtract in floating point. Unsigned integer losses would wrap around
        # and boolean losses cannot be subtracted.
        benchmark_arr = np.asarray(self.benchmark, dtype=float)
        models_arr = np.asarray(self.models, dtype=float)
        if not np.all(np.isfinite(benchmark_arr)):
            raise ValueError(
                "benchmark must be finite. The benchmark contains NaN or infinite "
                "values."
            )
        if not np.all(np.isfinite(models_arr)):
            raise ValueError(
                "models must be finite. The models contain NaN or infinite values."
            )
        self._loss_diff = benchmark_arr - models_arr
        # Constant, up to rounding error, so that the variance is 0
        constant = np.ptp(self._loss_diff, axis=0) <= (
            8 * np.finfo(float).eps * np.abs(self._loss_diff).max(axis=0)
        )
        if constant.any():
            constant_models = np.flatnonzero(constant).tolist()
            if isinstance(self.models, pd.DataFrame):
                constant_models = list(self.models.columns[constant])
            warnings.warn(
                "The loss differential has a variance of 0 for the models "
                f"{constant_models}, for example because a model is identical to "
                "the benchmark. The studentized test statistic of these models "
                "is not meaningful and the p-values of the SPA, and the superior "
                "models of StepM, are not reliable if they are included. Remove "
                "these models before testing.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._loss_diff_var = np.empty(0)
        self.t: int = self._loss_diff.shape[0]
        self.k: int = self._loss_diff.shape[1]
        bootstrap_cls = _bootstrap_class(bootstrap)
        bootstrap_inst = bootstrap_cls(self.block_size, self._loss_diff, seed=seed)
        self._seed = seed
        self._bootstrap = bootstrap_inst
        # Instantiate the kernel long-run variance estimator, which is used to
        # studentize the loss differentials and the bootstrap samples, and to
        # select the models to re-center. It depends on the bootstrap, which
        # cannot be replaced. A nested bootstrap replaces the kernel.
        self._kernel_variance: _KernelVariance | None = None
        if not self.nested:
            self._kernel_variance = _KernelVariance(self._bootstrap, self.t)
        self._pvalues: dict[str, float] = {}
        self._simulated_vals: Float64Array | None = None
        self._selector: BoolArray = np.ones(self.k, dtype=np.bool_)
        self._model = "SPA"
        if self.studentize:
            method = "bootstrap" if self.nested else "asymptotic"
        else:
            method = "none"
        self._info = {
            "studentization": method,
            "bootstrap": str(self.bootstrap),
            "ID": hex(id(self)),
        }

    def reset(self) -> None:
        """
        Reset the bootstrap to its initial state.
        """
        super().reset()
        self._pvalues = {}

    def subset(self, selector: BoolArray) -> None:
        """
        Set the active models to run the SPA on.

        This method is primarily for internal use.

        Parameters
        ----------
        selector : ndarray
            Boolean array with k elements indicating which columns to use when
            computing the p-values.  At least one element must be True. This is
            primarily for use by StepM.

        Raises
        ------
        TypeError
            If selector is not an array of Booleans.
        ValueError
            If selector does not have one element for each model or selects no
            model.
        """
        checked = check_array("selector", selector, shape=(self.k,))
        rule = f"a Boolean array with {self.k} elements and at least one True"
        if np.asarray(selector).dtype != np.bool_:
            raise TypeError(
                f"selector must be {rule}; got {np.asarray(selector).dtype} data "
                f"({type(selector).__name__})"
            )
        if not checked.any():
            raise ValueError(
                f"selector must be {rule}; got no True element "
                f"({type(selector).__name__})"
            )
        self._selector = np.asarray(selector)

    def compute(self) -> None:
        """
        Compute the bootstrap pvalue.

        Notes
        -----
        Must be called before accessing the pvalue.
        """
        # Plan
        # 1. Compute variances
        if self._simulated_vals is None:
            self._simulate_values()
        simulated_vals = self._simulated_vals
        # Use subset if needed
        assert simulated_vals is not None
        simulated_vals = simulated_vals[self._selector, :, :]
        max_simulated_vals = np.max(simulated_vals, 0)
        loss_diff_mean = self._studentized_mean()[self._selector]

        max_loss_diff = np.max(loss_diff_mean)
        pvalues = (max_simulated_vals > max_loss_diff).mean(axis=0)
        self._pvalues = {
            "lower": pvalues[0],
            "consistent": pvalues[1],
            "upper": pvalues[2],
        }

    def _simulate_values(self) -> None:
        self._compute_variance()
        # 2. Compute invalid columns using criteria for consistent
        self._valid_columns = self._check_column_validity()
        # 3. Compute simulated values
        # Upper always re-centers
        upper_mean = self._loss_diff.mean(axis=0)
        consistent_mean = upper_mean.copy()
        consistent_mean[np.logical_not(self._valid_columns)] = 0.0
        lower_mean = upper_mean.copy()
        # Lower does not re-center those that are worse
        lower_mean[lower_mean < 0] = 0.0
        means = [lower_mean, consistent_mean, upper_mean]
        simulated_vals = np.zeros((self.k, self.reps, 3))

        full_sample_scale = self._scale()
        for i, bs_data in enumerate(self.bootstrap.bootstrap(self.reps)):
            pos_arg, _ = bs_data
            loss_diff_star = np.asarray(pos_arg[0], dtype=float)
            # Studentize using the bootstrap sample, as in the statistic
            if self.studentize:
                nested_seed = None
                if self.nested:
                    # Seed the bootstrap of this sample from the generator, as
                    # when studentizing confidence intervals
                    nested_seed = int(
                        _get_random_integers(self.bootstrap.generator, 2**31 - 1)[0]
                    )
                variances = self._variance(loss_diff_star, nested_seed)
                scale = self._std_err(variances)
                # A column that is constant in the bootstrap sample, which
                # happens when a model differs from the benchmark in few
                # periods, has no standard error to studentize with
                degenerate = variances <= np.finfo(float).eps * self._loss_diff_var
                scale = np.where(degenerate, full_sample_scale, scale)
            else:
                scale = np.ones(self.k)
            loss_diff_star_mean = loss_diff_star.mean(axis=0)
            for j, mean in enumerate(means):
                simulated_vals[:, i, j] = (loss_diff_star_mean - mean) / scale
        self._simulated_vals = simulated_vals

    def _std_err(self, variances: Float64Array) -> Float64Array:
        """Standard errors of the mean loss differentials, bounded above zero"""
        std_err = np.sqrt(variances / self.t)
        return np.maximum(std_err, np.finfo(float).eps)

    def _scale(self) -> Float64Array:
        """Standard errors of the mean loss differentials if studentizing"""
        if not self.studentize:
            return np.ones(self.k)
        return self._std_err(self._loss_diff_var)

    def _studentized_mean(self) -> Float64Array:
        """Mean loss differentials, studentized if required"""
        return self._loss_diff.mean(axis=0) / self._scale()

    def _compute_variance(self) -> None:
        """
        Estimates the variance of the loss differentials

        Returns
        -------
        var : ndarray
            Array containing the variances of each loss differential
        """
        ld = self._loss_diff
        demeaned = ld - ld.mean(axis=0)
        variances = self._variance(demeaned, copy.deepcopy(self._seed))
        self._loss_diff_var = cast("np.ndarray", variances)

    def _variance(
        self,
        data: Float64Array,
        seed: int | np.random.Generator | np.random.RandomState | None = None,
    ) -> Float64Array:
        """
        Estimates the variance of the mean of each column of data

        Parameters
        ----------
        data : ndarray
            Data with t rows.
        seed : {int, Generator, RandomState}, optional
            Seed of the nested bootstrap. Not used unless nested is True.

        Returns
        -------
        ndarray
            The variance of the sample mean times t of each column, estimated
            using a bootstrap of the data if nested is True, and otherwise
            using the kernel estimator that matches the bootstrap.
        """
        if self.nested:
            bs = self.bootstrap.clone(data, seed=seed)
            means = bs.apply(lambda x: x.mean(axis=0), reps=self.reps)
            return self.t * means.var(axis=0)
        assert self._kernel_variance is not None
        return self._kernel_variance(data)

    def _check_column_validity(self) -> BoolArray:
        """
        Checks whether the loss from the model is too low relative to its mean
        to be asymptotically relevant.

        Returns
        -------
        valid : ndarray
            Boolean array indicating columns relevant for consistent p-value
            calculation
        """
        t, variances = self.t, self._loss_diff_var
        mean_loss_diff = self._loss_diff.mean(axis=0)
        threshold = -1.0 * np.sqrt((variances / t) * 2 * np.log(np.log(t)))
        return mean_loss_diff >= threshold

    @property
    def pvalues(self) -> pd.Series:
        """
        The lower, consistent and upper p-values.

        Returns
        -------
        pvals : Series
            Three p-values corresponding to the lower bound, the consistent
            estimator, and the upper bound.
        """
        self._check_compute()
        return pd.Series(list(self._pvalues.values()), index=list(self._pvalues.keys()))

    def critical_values(self, pvalue: float = 0.05) -> pd.Series:
        """
        Return data-dependent critical values.

        Parameters
        ----------
        pvalue : float, optional
            P-value in (0,1) to use when computing the critical values.

        Returns
        -------
        crit_vals : Series
            Series containing critical values for the lower, consistent and
            upper methodologies. When studentize is True, the critical values
            apply to the studentized mean loss differentials.

        Raises
        ------
        TypeError
            If pvalue is not a real number.
        ValueError
            If pvalue is not in (0,1).
        RuntimeError
            If compute has not been called.
        """
        pvalue = check_float(
            "pvalue", pvalue, lower=0.0, upper=1.0, inclusive="neither"
        )
        self._check_compute()
        # Subset if needed
        assert self._simulated_vals is not None
        simulated_values = self._simulated_vals[self._selector, :, :]
        max_simulated_values = np.max(simulated_values, axis=0)
        crit_vals = np.percentile(max_simulated_values, 100.0 * (1 - pvalue), axis=0)
        return pd.Series(crit_vals, index=list(self._pvalues.keys()))

    def better_models(
        self,
        pvalue: float = 0.05,
        pvalue_type: Literal["lower", "consistent", "upper"] = "consistent",
    ) -> Int64Array1D:
        """
        Return the set of models rejected as being equal-or-worse than the benchmark.

        Parameters
        ----------
        pvalue : float, optional
            P-value in (0,1) to use when computing superior models.
        pvalue_type : {"lower", "consistent", "upper"}, optional
            Indicates which critical value to use.

        Returns
        -------
        indices : ndarray
            Array containing the integer positions of the columns of the
            superior models. Positions are returned, and not column names, even
            if models is a DataFrame.

        Raises
        ------
        TypeError
            If pvalue is not a real number or pvalue_type is not a string.
        ValueError
            If pvalue is not in (0,1) or pvalue_type is not one of the allowed
            values.
        RuntimeError
            If compute has not been called.

        Notes
        -----
        List of superior models returned is always with respect to the initial
        set of models, even when using subset().
        """
        pvalue = check_float(
            "pvalue", pvalue, lower=0.0, upper=1.0, inclusive="neither"
        )
        pvalue_key = check_option("pvalue_type", pvalue_type, _PVALUE_TYPES)
        self._check_compute()
        crit_val = self.critical_values(pvalue=pvalue)[pvalue_key]
        better_models = self._studentized_mean() > crit_val
        better_models = np.logical_and(better_models, self._selector)
        return np.argwhere(better_models).flatten()

    def _check_compute(self) -> None:
        if self._pvalues:
            return
        msg = "compute must be called before pvalues are available."
        raise RuntimeError(msg)


class RealityCheck(SPA):
    # Shallow clone of SPA
    pass
