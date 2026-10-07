from collections.abc import Hashable, Sequence
import copy
from typing import cast
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
)
from arch.utility.array import DocStringInheritor, ensure2d

__all__ = ["MCS", "SPA", "RealityCheck", "StepM"]


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
    Abstract class for inheritance
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
        The bootstrap used in the comparison

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
        Reset the bootstrap to it's initial state.
        """
        self.bootstrap.reset()


class MCS(MultipleComparison):
    """
    Model Confidence Set (MCS) of Hansen, Lunde and Nason.

    Parameters
    ----------
    losses : {ndarray, DataFrame}
        T by k array containing losses from a set of models
    size : float, optional
        Value in (0,1) to use as the test size when implementing the
        mcs. Default value is 0.05.
    block_size : int, optional
        Length of window to use in the bootstrap.  If not provided, sqrt(T)
        is used.  In general, this should be provided and chosen to be
        appropriate for the data.
    method : {'max', 'R'}, optional
        MCS test and elimination implementation method, either 'max' or 'R'.
        Default is 'R'.
    reps : int, optional
        Number of bootstrap replications to uses.  Default is 1000.
    bootstrap : str, optional
        Bootstrap to use.  Options are
        'stationary' or 'sb': Stationary bootstrap (Default)
        'circular' or 'cbb': Circular block bootstrap
        'moving block' or 'mbb': Moving block bootstrap
    seed : {int, Generator, RandomState}, optional
        Seed value to use when creating the bootstrap used in the comparison.
        If an integer or None, the NumPy default_rng is used with the seed
        value.  If a Generator or a RandomState, the argument is used.

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
        self.losses = ensure2d(losses, "losses")
        self._losses_arr = np.asarray(self.losses, dtype=float)
        if self._losses_arr.shape[1] < 2:
            raise ValueError("losses must have at least two columns")
        self.size: float = size
        self.reps: int = reps
        if block_size is None:
            self.block_size = int(np.sqrt(losses.shape[0]))
        else:
            self.block_size = block_size

        self.t: int = losses.shape[0]
        self.k: int = losses.shape[1]
        self.method: Literal["R", "max"] = method
        # Bootstrap indices since the same bootstrap should be used in the
        # repeated steps
        indices = np.arange(self.t)
        bootstrap_meth = bootstrap.lower().replace(" ", "_")
        if bootstrap_meth in ("circular", "cbb"):
            bootstrap_inst = CircularBlockBootstrap(self.block_size, indices, seed=seed)
        elif bootstrap_meth in ("stationary", "sb"):
            bootstrap_inst = StationaryBootstrap(self.block_size, indices, seed=seed)
        elif bootstrap_meth in ("moving_block", "mbb"):
            bootstrap_inst = MovingBlockBootstrap(self.block_size, indices, seed=seed)
        else:
            raise ValueError(f"Unknown bootstrap: {bootstrap_meth}")
        self._seed = seed
        self._bootstrap = bootstrap_inst
        self._bootstrap_indices: list[IntArray] = []  # For testing
        self._model = "MCS"
        self._info = {
            "size": f"{self.size:0.2f}",
            "bootstrap": str(bootstrap_inst),
            "ID": hex(id(self)),
        }
        self._results_computed = False

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
        if self.method.lower() == "r":
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
        for j, data in enumerate(bs.bootstrap(self.reps)):
            bs_index = data[0][0]  # Only element in pos data
            self._bootstrap_indices.append(
                np.asarray(bs_index, dtype=int)
            )  # For testing
            mean_losses_star = losses[bs_index].mean(axis=0)[:, None]
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
            # Loc has indices i,j, -- i  is the elimination
            # Diffs are L(i) - L(j), so large value in [i,j] indicates
            # i is worse than j
            # Elimination is for
            i = loc.squeeze()[0]
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
        for i, data in enumerate(self.bootstrap.bootstrap(self.reps)):
            bs_index = data[0][0]
            self._bootstrap_indices.append(
                np.asarray(bs_index, dtype=int)
            )  # For testing
            bs_errors = loss_errors[bs_index]
            avg_bs_errors = bs_errors.mean(axis=0)
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
        List of model indices that are included in the MCS

        Returns
        -------
        included : list
            List of column indices or names of the included models
        """
        self._has_been_computed()
        incl_loc = self._pvalues.Pvalue > self.size
        included = list(self._pvalues.index[incl_loc])
        included.sort()
        return included

    @property
    def excluded(self) -> list[Hashable]:
        """
        List of model indices that are excluded from the MCS

        Returns
        -------
        excluded : list
            List of column indices or names of the excluded models
        """
        self._has_been_computed()
        excl_loc = self._pvalues.Pvalue <= self.size
        excluded = list(self._pvalues.index[excl_loc])
        excluded.sort()
        return excluded

    @property
    def pvalues(self) -> pd.DataFrame:
        """
        Model p-values for inclusion in the MCS

        Returns
        -------
        pvalues : DataFrame
            DataFrame where the index is the model index (column or name)
            containing the smallest size where the model is in the MCS.
        """
        self._has_been_computed()
        return self._pvalues


class StepM(MultipleComparison):
    """
    StepM multiple comparison procedure of Romano and Wolf.

    Parameters
    ----------
    benchmark : {ndarray, Series}
        T element array of benchmark model *losses*
    models : {ndarray, DataFrame}
        T by k element array of alternative model *losses*
    size : float, optional
        Value in (0,1) to use as the test size when implementing the
        comparison. Default value is 0.05.
    block_size : int, optional
        Length of window to use in the bootstrap.  If not provided, sqrt(T)
        is used.  In general, this should be provided and chosen to be
        appropriate for the data.
    reps : int, optional
        Number of bootstrap replications to uses.  Default is 1000.
    bootstrap : str, optional
        Bootstrap to use.  Options are
        'stationary' or 'sb': Stationary bootstrap (Default)
        'circular' or 'cbb': Circular block bootstrap
        'moving block' or 'mbb': Moving block bootstrap
    studentize : bool, optional
        Flag indicating to studentize loss differentials. Default is True.
        Studentization is also applied inside each bootstrap replication,
        see :class:`~arch.bootstrap.SPA`.
    nested : bool, optional
        Flag indicating to use a nested bootstrap to compute the variances
        of the loss differentials in the original data. Default is False,
        which uses a kernel estimator of the long-run variance that matches
        the bootstrap. Note that this can be slow since the procedure requires
        an extra bootstrap. Studentization inside each bootstrap replication
        always uses the kernel estimator.
    seed : {int, Generator, RandomState}, optional
        Seed value to use when creating the bootstrap used in the comparison.
        If an integer or None, the NumPy default_rng is used with the seed
        value.  If a Generator or a RandomState, the argument is used.

    Notes
    -----
    The size controls the Family Wise Error Rate (FWER) since this is a
    multiple comparison procedure.  Uses SPA and the consistent selection
    procedure.

    See :class:`~arch.bootstrap.SPA` for a description of how loss
    differentials are studentized.

    See [1]_ for detail.

    See Also
    --------
    SPA

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
        self.benchmark = ensure2d(benchmark, "benchmark")
        self.models = ensure2d(models, "models")
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
        self.block_size: int = self.spa.block_size
        self.t: int = self.models.shape[0]
        self.k: int = self.models.shape[1]
        self.reps: int = reps
        self.size: float = size
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
        List of the indices or column names of the superior models

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
    benchmark : {ndarray, Series}
        T element array of benchmark model *losses*
    models : {ndarray, DataFrame}
        T  by k element array of alternative model *losses*
    block_size : int, optional
        Length of window to use in the bootstrap.  If not provided, sqrt(T)
        is used.  In general, this should be provided and chosen to be
        appropriate for the data.
    reps : int, optional
        Number of bootstrap replications to uses.  Default is 1000.
    bootstrap : str, optional
        Bootstrap to use.  Options are
        'stationary' or 'sb': Stationary bootstrap (Default)
        'circular' or 'cbb': Circular block bootstrap
        'moving block' or 'mbb': Moving block bootstrap
    studentize : bool
        Flag indicating to studentize loss differentials. Default is True.
        Studentization is also applied inside each bootstrap replication,
        see Notes.
    nested : bool
        Flag indicating to use a nested bootstrap to compute the variances
        of the loss differentials in the original data. Default is False,
        which uses a kernel estimator of the long-run variance that matches
        the bootstrap. Note that this can be slow since the procedure requires
        an extra bootstrap. Studentization inside each bootstrap replication
        always uses the kernel estimator since a nested bootstrap in each
        replication would be prohibitively slow.
    seed : {int, Generator, RandomState}, optional
        Seed value to use when creating the bootstrap used in the comparison.
        If an integer or None, the NumPy default_rng is used with the seed
        value.  If a Generator or a RandomState, the argument is used.

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

    The standard errors are the square root of a kernel estimator of the
    long-run variance divided by T. The kernel is chosen to match the
    bootstrap, so that the estimator is the variance of the sample mean that
    is implied by the bootstrap, using the average block size as the
    bandwidth.

        - Stationary bootstrap : The lag :math:`i` autocovariance has weight
          :math:`(1-i/T)(1-p)^i + (i/T)(1-p)^{T-i}` where :math:`p` is the
          inverse of the block size [4]_.
        - Circular block bootstrap : The Bartlett kernel applied to the
          circular autocovariances.
        - Moving block bootstrap : The Bartlett kernel.

    See [1]_ and [2]_ for details.

    See Also
    --------
    StepM

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
        super().__init__()
        self.benchmark = ensure2d(benchmark, "benchmark")
        self.models = ensure2d(models, "models")
        self.reps: int = reps
        if block_size is None:
            self.block_size = int(np.sqrt(benchmark.shape[0]))
        else:
            self.block_size = block_size
        self.studentize: bool = studentize
        self.nested: bool = nested
        self._loss_diff = np.asarray(self.benchmark) - np.asarray(self.models)
        self._loss_diff_var = np.empty(0)
        self.t: int = self._loss_diff.shape[0]
        self.k: int = self._loss_diff.shape[1]
        bootstrap_name = bootstrap.lower().replace(" ", "_")
        if bootstrap_name in ("circular", "cbb"):
            bootstrap_inst = CircularBlockBootstrap(
                self.block_size, self._loss_diff, seed=seed
            )
        elif bootstrap_name in ("stationary", "sb"):
            bootstrap_inst = StationaryBootstrap(
                self.block_size, self._loss_diff, seed=seed
            )
        elif bootstrap_name in ("moving_block", "mbb"):
            bootstrap_inst = MovingBlockBootstrap(
                self.block_size, self._loss_diff, seed=seed
            )
        else:
            raise ValueError(f"Unknown bootstrap: {bootstrap_name}")
        self._seed = seed
        self._bootstrap = bootstrap_inst
        # Instantiate the long-run variance estimator used to studentize the
        # loss differentials and the bootstrap samples. It depends on the
        # bootstrap, which cannot be replaced.
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
        Sets a list of active models to run the SPA on.  Primarily for
        internal use.

        Parameters
        ----------
        selector : ndarray
            Boolean array indicating which columns to use when computing the
            p-values.  This is primarily for use by StepM.
        """
        self._selector = selector

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
                variances = self._kernel_variance(loss_diff_star)
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
        if self.nested:
            # Use bootstrap to estimate variances
            bs = self.bootstrap.clone(demeaned, seed=copy.deepcopy(self._seed))
            means = bs.apply(lambda x: x.mean(axis=0), reps=self.reps)
            variances = self.t * means.var(axis=0)
        else:
            variances = self._kernel_variance(demeaned)
        self._loss_diff_var = cast("np.ndarray", variances)

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
        P-values corresponding to the lower, consistent and
        upper p-values.

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
        Returns data-dependent critical values

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
        """
        self._check_compute()
        if not (0.0 < pvalue < 1.0):
            raise ValueError("pvalue must be in (0,1)")
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
        Returns set of models rejected as being equal-or-worse than the
        benchmark

        Parameters
        ----------
        pvalue : float, optional
            P-value in (0,1) to use when computing superior models
        pvalue_type : str, optional
            String in 'lower', 'consistent', or 'upper' indicating which
            critical value to use.

        Returns
        -------
        indices : list
            List of column names or indices of the superior models.  Column
            names are returned if models is a DataFrame.

        Notes
        -----
        List of superior models returned is always with respect to the initial
        set of models, even when using subset().
        """
        self._check_compute()
        if pvalue_type not in self._pvalues:
            raise ValueError("Unknown pvalue type")
        crit_val = self.critical_values(pvalue=pvalue)[pvalue_type]
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
