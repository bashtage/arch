from __future__ import division
from ..compat.python import add_metaclass, iteritems
from ..utility.array import DocStringInheritor, ensure2d

from collections import OrderedDict

import numpy as np
import pandas as pd

from .base import StationaryBootstrap, CircularBlockBootstrap, \
    MovingBlockBootstrap

__all__ = ['StepM', 'SPA', 'RealityCheck']


def _info_to_str(model, info, is_repr=False, is_html=False):
    if is_html:
        model = '<strong>' + model + '</strong>'
    _str = model + '('
    for k, v in iteritems(info):
        if k.lower() != 'id' or is_repr:
            if is_html:
                k = '<strong>' + k + '</strong>'
            _str += k + ': ' + v + ', '
    return _str[:-2] + ')'


class MultipleComparison(object):
    """
    Abstract class for inheritance
    """

    def __init__(self):
        self._model = ''
        self._info = OrderedDict()
        self.bootstrap = None

    def __str__(self):
        return _info_to_str(self._model, self._info, False)

    def __repr__(self):
        return _info_to_str(self._model, self._info, True)

    def _repr_html_(self):
        return _info_to_str(self._model, self._info, True, True)

    def reset(self):
        """
        Reset the bootstrap to it's initial state.
        """
        self.bootstrap.reset()

    def seed(self, value):
        """
        Seeds the bootstrap's random number generator

        Parameters
        ----------
        value : int
            Integer to use as the seed
        """
        self.bootstrap.seed(value)


class MCS(MultipleComparison):
    """
    Implementation of the Model Confidence Set (MCS)

    Parameters
    ----------
    losses : array-like
        T by k array containing losses from a set of models
    size : float, optional
        Value in (0,1) to use as the test size when implementing the
        mcs. Default value is 0.05.
    block_size : int, optional
        Length of window to use in the bootstrap.  If not provided, sqrt(T)
        is used.  In general, this should be provided and chosen to be
        appropriate for the data.
    method : str, optional
        MCS test and elimination implementation method: either 'max' or 'R'.
        Default is 'R'.
    reps : int, optional
        Number of bootstrap replications to uses.  Default is 1000.
    bootstrap : str, optional
        Bootstrap to use.  Options are
        'stationary' or 'sb': Stationary bootstrap (Default)
        'circular' or 'cbb': Circular block bootstrap
        'moving block' or 'mbb': Moving block bootstrap


    Attributes
    ----------
    pvalues : DataFrame
        DataFrame where the index is the model index (column or name)
        containing the smallest size where the model is in the MCS.
    included : list
        List of column indices or names of the included models
    excluded : list
        List of column indices or names of the excluded models

    Methods
    -------
    compute

    References
    ----------
    Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
    Econometrica, 79(2), 453-497.
    """

    def __init__(self, losses, size, reps=1000, block_size=None, method='R',
                 bootstrap='stationary'):
        super(MCS, self).__init__()
        self.losses = ensure2d(losses, 'losses')
        self._losses_arr = np.asarray(self.losses)
        if self._losses_arr.shape[1] < 2:
            raise ValueError('losses must have at least two columns')
        self.size = size
        self.reps = reps
        if block_size is None:
            self.block_size = int(np.sqrt(losses.shape[0]))
        else:
            self.block_size = block_size

        self.t, self.k = losses.shape
        self.method = method
        # Bootstrap indices since the same bootstrap should be used in the
        # repeated steps
        indices = np.arange(self.t)
        bootstrap = bootstrap.lower().replace(' ', '_')
        if bootstrap in ('stationary', 'sb'):
            bootstrap = StationaryBootstrap(self.block_size, indices)
        elif bootstrap in ('circular', 'cbb'):
            bootstrap = CircularBlockBootstrap(self.block_size, indices)
        elif bootstrap in ('moving_block', 'mbb'):
            bootstrap = MovingBlockBootstrap(self.block_size, indices)
        else:
            raise ValueError('Unknown bootstrap:' + bootstrap)
        self.bootstrap = bootstrap
        self._bootsrap_indices = []  # For testing
        self._model = 'MCS'
        self._info = OrderedDict([('size', '{0:0.2f}'.format(self.size)),
                                  ('bootstrap', str(bootstrap)),
                                  ('ID', hex(id(self)))])

    def _format_pvalues(self, eliminated):
        columns = ['Model index', 'Pvalue']
        mcs = pd.DataFrame(eliminated, columns=columns)
        max_pval = mcs.iloc[0, 1]
        for i in range(1, mcs.shape[0]):
            max_pval = np.max([max_pval, mcs.iloc[i, 1]])
            mcs.iloc[i, 1] = max_pval
        model_index = mcs.pop('Model index')
        if isinstance(self.losses, pd.DataFrame):
            # Workaround for old pandas/numpy combination
            # Preferred expression :
            # model_index = pd.Series(self.losses.columns[model_index])
            model_index = self.losses.iloc[:, model_index.values].columns
            model_index = pd.Series(model_index)
            model_index.name = 'Model name'
        mcs.index = model_index
        return mcs

    def compute(self):
        """
        Computes the model confidence set
        """
        if self.method.lower() == 'r':
            self._compute_r()
        else:
            self._compute_max()

    def _compute_r(self):
        """
        Computes the model confidence set using the R method
        """
        # R method
        # 1. Compute pairwise difference (k,k)
        losses = self._losses_arr
        mean_losses = losses.mean(0)[:, None]
        loss_diffs = mean_losses - mean_losses.T
        # Compute pairwise variance using bootstrap (k,k)
        # In each bootstrap, save the average difference of each pair (b,k,k)
        bootstrapped_mean_losses = np.zeros((self.reps, self.k, self.k))
        bs = self.bootstrap
        for j, data in enumerate(bs.bootstrap(self.reps)):
            bs_index = data[0][0]  # Only element in pos data
            self._bootsrap_indices.append(bs_index)  # For testing
            mean_losses_star = losses[bs_index].mean(0)[:, None]
            bootstrapped_mean_losses[j] = mean_losses_star - mean_losses_star.T
        # Recenter
        bootstrapped_mean_losses -= loss_diffs
        variances = (bootstrapped_mean_losses ** 2).mean(0)
        variances += np.eye(self.k)  # Prevent division by 0
        self._variances = variances
        # Standardize everything
        std_loss_diffs = loss_diffs / np.sqrt(variances)
        std_bs_mean_losses = bootstrapped_mean_losses / np.sqrt(variances)
        # 3. Using the models still in the set, compute the max (b,1)
        # Initialize the set
        included = np.ones(self.k, dtype=np.bool)
        # Loop until there is only 1 model left
        eliminated = []
        while included.sum() > 1:
            indices = np.argwhere(included)
            included_loss_diffs = std_loss_diffs[indices, indices.T]
            test_stat = np.max(included_loss_diffs)
            included_bs_losses = std_bs_mean_losses[:, indices, indices.T]
            simulated_test_stat = np.max(np.max(included_bs_losses, 2), 1)
            pval = (test_stat < simulated_test_stat).mean()
            loc = np.argwhere(included_loss_diffs == test_stat)
            # Loc has indices i,j, -- i  is the elimination
            # Diffs are L(i) - L(j), so large value in [i,j] indicates
            # i is worse than j
            # Elimination is for
            i = loc.squeeze()[0]
            eliminated.append([indices.flat[i], pval])
            included[indices.flat[i]] = False
        # Add pval of 1 for model remaining
        indices = np.argwhere(included).flatten()
        for ind in indices:
            eliminated.append([ind, 1.0])
        self._pvalues = self._format_pvalues(eliminated)

    def _compute_max(self):
        """
        Computes the model confidence set using the R method
        """
        # max method
        losses = self._losses_arr
        # 1. compute loss "errors"
        loss_errors = losses - losses.mean(0)
        # Generate bootstrap samples
        bs_avg_loss_errors = np.zeros((self.reps, self.k))
        for i, data in enumerate(self.bootstrap.bootstrap(self.reps)):
            bs_index = data[0][0]
            self._bootsrap_indices.append(bs_index)  # For testing
            bs_errors = loss_errors[bs_index]
            avg_bs_errors = bs_errors.mean(0)
            avg_bs_errors -= avg_bs_errors.mean()
            bs_avg_loss_errors[i] = avg_bs_errors
            # Initialize the set
        included = np.ones(self.k, dtype=np.bool)
        # Loop until there is only 1 model left
        eliminated = []
        while included.sum() > 1:
            indices = np.argwhere(included)
            incl_losses = losses[:, included]
            incl_bs_avg_loss_err = bs_avg_loss_errors[:, included]
            incl_bs_grand_loss = incl_bs_avg_loss_err.mean(1)
            # Reshape for broadcast
            incl_bs_avg_loss_err -= incl_bs_grand_loss[:, None]
            std_devs = np.sqrt((incl_bs_avg_loss_err ** 2).mean(0))
            simulated_test_stat = incl_bs_avg_loss_err / std_devs
            simulated_test_stat = np.max(simulated_test_stat, 1)
            loss_diffs = incl_losses.mean(0)
            loss_diffs -= loss_diffs.mean()
            std_loss_diffs = loss_diffs / std_devs
            test_stat = np.max(std_loss_diffs)
            pval = (test_stat < simulated_test_stat).mean()
            i = np.argwhere(std_loss_diffs == test_stat)
            eliminated.append([indices.flat[i.squeeze()], pval])
            included[indices.flat[i]] = False

        self._pvalues = self._format_pvalues(eliminated)

    @property
    def included(self):
        """
        Returns a list of model indices that are included in the MCS
        """
        included = (self._pvalues.Pvalue > self.size)
        included = list(self._pvalues.index[included])
        included.sort()
        return included

    @property
    def excluded(self):
        """
        Returns a list of model indices that are excluded from the MCS
        """
        excluded = (self._pvalues.Pvalue <= self.size)
        excluded = list(self._pvalues.index[excluded])
        excluded.sort()
        return excluded

    @property
    def pvalues(self):
        """
        Returns a DataFrame containing model index and the smallest size
        where it is in the MCS
        """
        return self._pvalues


class StepM(MultipleComparison):
    """
    Implementation of Romano and Wolf's StepM multiple comparison procedure

    Parameters
    ----------
    benchmark : array-like
        T element array of benchmark model *losses*
    models : array-like
        T  by k element array of alternative model *losses*
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
        Flag indicating to studentize loss differentials. Default is True
    nested : bool, optional
        Flag indicating to use a nested bootstrap to compute variances for
        studentization.  Default is False.  Note that this can be slow since
        the procedure requires k extra bootstraps.

    Methods
    -------
    compute
        Compute the set of superior models.

    Attributes
    ----------
    superior_models : list
        List of superior models.  Contains column indices if models is an
        array or contains column names if models is a DataFrame.

    References
    ----------
    Romano, J. P., & Wolf, M. (2005). "Stepwise multiple testing as formalized
    data snooping." Econometrica, 73(4), 1237-1282.

    Notes
    -----
    The size controls the Family Wise Error Rate (FWER) since this is a
    multiple comparison procedure.  Uses SPA and the consistent selection
    procedure.

    See Also
    --------
    SPA
    """

    def __init__(self, benchmark, models, size=0.05, block_size=None,
                 reps=1000, bootstrap='stationary', studentize=True,
                 nested=False):
        super(StepM, self).__init__()
        self.benchmark = ensure2d(benchmark, 'benchmark')
        self.models = ensure2d(models, 'models')
        self.spa = SPA(benchmark, models, block_size=block_size, reps=reps,
                       bootstrap=bootstrap,
                       studentize=studentize, nested=nested)
        self.block_size = self.spa.block_size
        self.t, self.k = self.models.shape
        self.reps = reps
        self.size = size
        self._superior_models = None
        self.bootstrap = self.spa.bootstrap

        self._model = 'StepM'
        if self.spa.studentize:
            method = 'bootstrap' if self.spa.nested else 'asymptotic'
        else:
            method = 'none'
        self._info = OrderedDict([('FWER (size)', '{:0.2f}'.format(self.size)),
                                  ('studentization', method),
                                  ('bootstrap', str(self.spa.bootstrap)),
                                  ('ID', hex(id(self)))])

    def compute(self):
        """
        Computes the set of superior models
        """
        # 1. Run SPA
        self.spa.compute()
        # 2. If any models superior, store indices, remove and re-run SPA
        better_models = list(self.spa.better_models(self.size))
        all_better_models = better_models
        # 3. Stop if nothing superior
        while better_models and (len(better_models) < self.k):
            # A. Use Selector to remove better models
            selector = np.ones(self.k, dtype=np.bool)
            if isinstance(self.models, pd.DataFrame):  # Columns
                selector[self.models.columns.isin(all_better_models)] = False
            else:
                selector[np.array(list(all_better_models))] = False
            self.spa.subset(selector)
            # B. Rerun
            self.spa.compute()
            better_models = list(self.spa.better_models(self.size))
            all_better_models.extend(better_models)
        # Reset SPA
        selector = np.ones(self.k, dtype=np.bool)
        self.spa.subset(selector)
        all_better_models = list(all_better_models)
        all_better_models.sort()
        self._superior_models = all_better_models

    @property
    def superior_models(self):
        """
        Returns a list of the indices or column names of the superior models.
        """
        if self._superior_models is None:
            msg = 'compute must be callded before accessing superior_models'
            raise RuntimeError(msg)
        return self._superior_models


@add_metaclass(DocStringInheritor)
class SPA(MultipleComparison):
    """
    Implementation of the Test of Superior Predictive Ability (SPA),
    which is also known as the Reality Check or Bootstrap Data Snooper.

    Parameters
    ----------
    benchmark : array-like
        T element array of benchmark model *losses*
    models : array-like
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
        Flag indicating to studentize loss differentials. Default is True
    nested=False
        Flag indicating to use a nested bootstrap to compute variances for
        studentization.  Default is False.  Note that this can be slow since
        the procedure requires k extra bootstraps.

    Methods
    -------
    compute
        Compute the bootstrap pvalue.  Must be called before accessing the
        pvalue
    seed
        Pass seed to bootstrap implementation
    reset
        Reset the bootstrap to its initial state
    better_models
        Produce a list of column indices or names (if models is a DataFrame)
        that are rejected given a test size

    Attributes
    ----------
    pvalues : Series
        A set of three p-values corresponding to the lower, consistent and
        upper p-values.

    References
    ----------
    White, H. (2000). "A reality check for data snooping." Econometrica 68,
    no. 5, 1097-1126.

    Hansen, P. R. (2005). "A test for superior predictive ability."
    Journal of Business & Economic Statistics, 23(4)

    Notes
    -----
    The three p-value correspond to different re-centering decisions.
        - Upper : Never recenter to all models are relevant to distribution
        - Consistent : Only recenter if closer than a log(log(t)) bound
        - Lower : Never recenter a model if worse than benchmark

    See Also
    --------
    StepM

    """

    def __init__(self, benchmark, models, block_size=None, reps=1000,
                 bootstrap='stationary', studentize=True, nested=False):
        super(SPA, self).__init__()
        self.benchmark = ensure2d(benchmark, 'benchmark')
        self.models = ensure2d(models, 'models')
        self.reps = reps
        if block_size is None:
            self.block_size = int(np.sqrt(benchmark.shape[0]))
        else:
            self.block_size = block_size
        self.studentize = studentize
        self.nested = nested
        self._loss_diff = np.asarray(self.benchmark) - np.asarray(self.models)
        self._loss_diff_var = None
        self.t, self.k = self._loss_diff.shape
        bootstrap = bootstrap.lower().replace(' ', '_')
        if bootstrap in ('stationary', 'sb'):
            bootstrap = StationaryBootstrap(self.block_size, self._loss_diff)
        elif bootstrap in ('circular', 'cbb'):
            bootstrap = CircularBlockBootstrap(self.block_size,
                                               self._loss_diff)
        elif bootstrap in ('moving_block', 'mbb'):
            bootstrap = MovingBlockBootstrap(self.block_size, self._loss_diff)
        else:
            raise ValueError('Unknown bootstrap:' + bootstrap)
        self.bootstrap = bootstrap
        self._pvalues = None
        self._simulated_vals = None
        self._selector = np.ones(self.k, dtype=np.bool)
        self._model = 'SPA'
        if self.studentize:
            method = 'bootstrap' if self.nested else 'asymptotic'
        else:
            method = 'none'
        self._info = OrderedDict([('studentization', method),
                                  ('bootstrap', str(self.bootstrap)),
                                  ('ID', hex(id(self)))])

    def reset(self):
        """
        Reset the bootstrap to it's initial state.
        """
        super(SPA, self).reset()
        self._pvalues = None

    def subset(self, selector):
        """
        Sets a list of active models to run the SPA on.  Primarily for
        internal use.

        Parameters
        ----------
        selector : array
            Boolean array indicating which columns to use when computing the
            p-values.  This is primarily for use by StepM.
        """
        self._selector = selector

    def compute(self):
        """
        Compute the bootstrap p-value

        """
        # Plan
        # 1. Compute variances
        if self._simulated_vals is None:
            self._simulate_values()
        simulated_vals = self._simulated_vals
        # Use subset if needed
        simulated_vals = simulated_vals[self._selector, :, :]
        max_simulated_vals = np.max(simulated_vals, 0)
        loss_diff = self._loss_diff[:, self._selector]

        max_loss_diff = np.max(loss_diff.mean(axis=0))
        pvalues = (max_simulated_vals > max_loss_diff).mean(axis=0)
        self._pvalues = OrderedDict([('lower', pvalues[0]),
                                     ('consistent', pvalues[1]),
                                     ('upper', pvalues[2])])

    def _simulate_values(self):
        self._compute_variance()
        # 2. Compute invalid columns using criteria for consistent
        self._valid_columns = self._check_column_validity()
        # 3. Compute simulated values
        # Upper always re-centers
        upper_mean = self._loss_diff.mean(0)
        consistent_mean = upper_mean.copy()
        consistent_mean[np.logical_not(self._valid_columns)] = 0.0
        lower_mean = upper_mean.copy()
        # Lower does not re-center those that are worse
        lower_mean[lower_mean < 0] = 0.0
        means = [lower_mean, consistent_mean, upper_mean]
        simulated_vals = np.zeros((self.k, self.reps, 3))
        for i, bsdata in enumerate(self.bootstrap.bootstrap(self.reps)):
            pos_arg, kw_arg = bsdata
            loss_diff_star = pos_arg[0]
            for j, mean in enumerate(means):
                simulated_vals[:, i, j] = loss_diff_star.mean(0) - mean
        self._simulated_vals = np.array(simulated_vals)

    def _compute_variance(self):
        """
        Estimates the variance of the loss differentials

        Returns
        -------
        var : array
            Array containing the variances of each loss differential
        """
        ld = self._loss_diff
        demeaned = ld - ld.mean(axis=0)
        if self.nested:
            # Use bootstrap to estimate variances
            bs = self.bootstrap.clone(demeaned)
            means = bs.apply(lambda x: x.mean(0), reps=self.reps)
            variances = self.t * means.var(axis=0)
        else:
            t = self.t
            p = 1.0 / self.block_size
            variances = np.sum(demeaned ** 2, 0) / t
            for i in range(1, t):
                kappa = ((1.0 - (i / t)) * ((1 - p) ** i)) + (
                    (i / t) * ((1 - p) ** (t - i)))
                variances += 2 * kappa * np.sum(
                    demeaned[:(t - i), :] * demeaned[i:, :], 0) / t
        self._loss_diff_var = variances

    def _check_column_validity(self):
        """
        Checks whether the loss from the model is too low relative to its mean
        to be asymptotically relevant.

        Returns
        -------
        valid : array
            Boolean array indicating columns relevant for consistent p-value
            calculation
        """
        t, variances = self.t, self._loss_diff_var
        mean_loss_diff = self._loss_diff.mean(0)
        threshold = -1.0 * np.sqrt((variances / t) * 2 * np.log(np.log(t)))
        return mean_loss_diff >= threshold

    @property
    def pvalues(self):
        """
        Returns Series containing three p-values corresponding to the
        lower, consistent and upper methodologies.
        """
        self._check_compute()
        return pd.Series(list(self._pvalues.values()),
                         index=list(self._pvalues.keys()))

    def critical_values(self, pvalue=0.05):
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
            upper methodologies
        """
        self._check_compute()
        if not (0.0 < pvalue < 1.0):
            raise ValueError('pvalue must be in (0,1)')
        # Subset if neded
        simulated_values = self._simulated_vals[self._selector, :, :]
        max_simulated_values = np.max(simulated_values, axis=0)
        crit_vals = np.percentile(max_simulated_values,
                                  100.0 * (1 - pvalue),
                                  axis=0)
        return pd.Series(crit_vals, index=list(self._pvalues.keys()))

    def better_models(self, pvalue=0.05, pvalue_type='consistent'):
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
            raise ValueError('Unknown pvalue type')
        crit_val = self.critical_values(pvalue=pvalue)[pvalue_type]
        better_models = self._loss_diff.mean(0) > crit_val
        better_models = np.logical_and(better_models, self._selector)
        if isinstance(self.models, pd.DataFrame):
            return list(self.models.columns[better_models])
        else:
            return np.argwhere(better_models).flatten()

    def _check_compute(self):
        if self._pvalues is not None:
            return None
        msg = 'compute must be called before pvalues are available.'
        raise RuntimeError(msg)


class RealityCheck(SPA):
    # Shallow clone of SPA
    pass
