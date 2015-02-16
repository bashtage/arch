from __future__ import division

from collections import OrderedDict

import numpy as np
import pandas as pd

from .base import StationaryBootstrap, CircularBlockBootstrap, MovingBlockBootstrap
from arch.compat.python import add_metaclass
from arch.utils import DocStringInheritor, ensure2d

__all__ = ['StepM', 'SPA', 'RealityCheck']


class StepM(object):
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
    Romano, J. P., Shaikh, A.M., and Wolf. M. (2008) "Formalized data snooping
    based on generalized error rates." Econometric Theory 24, no. 02. 404-447.

    Notes
    -----
    The size controls the Family Wise Error Rate (FWER) since this is a
    multiple comparison procedure.  Uses SPA and the consistent selection
    procedure.

    See Also
    --------
    SPA
    """
    def __init__(self, benchmark, models, size=0.05, block_size=None, reps=1000,
                 bootstrap='stationary', studentize=True, nested=False):
        self.benchmark = ensure2d(benchmark, 'benchmark')
        self.models = ensure2d(models, 'models')
        self.spa = SPA(benchmark, models, block_size=block_size, reps=reps, bootstrap=bootstrap,
                       studentize=studentize, nested=nested)
        self.block_size = self.spa.block_size
        self.t, self.k = self.models.shape
        self.reps = reps
        self.size = size
        self._superior_models = None

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
        while better_models:
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
        return self._superior_models


@add_metaclass(DocStringInheritor)
class SPA(object):
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

    Hansen, P. R.. (2005). "A test for superior predictive ability."
    Journal of Business & Economic Statistics 23, no. 4.

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
        self.benchmark = ensure2d(benchmark, 'benchmark')
        self.models = ensure2d(models, 'models')
        self.reps = reps
        if block_size is None:
            self.block_size = int(np.sqrt(benchmark.shape[0]))
        else:
            self.block_size = block_size
        self.studentize = studentize
        self.nested = nested
        self._loss_diff = np.asarray(models) - np.asarray(self.benchmark)
        self._loss_diff_var = None
        self.t, self.k = self._loss_diff.shape
        if bootstrap in ('stationary', 'sb'):
            bootstrap = StationaryBootstrap(self.block_size, self._loss_diff)
        elif bootstrap in ('circular', 'cbb'):
            bootstrap = CircularBlockBootstrap(self.block_size,
                                               self._loss_diff)
        elif bootstrap in ('moving_block', 'mbb'):
            bootstrap = MovingBlockBootstrap(self.block_size, self._loss_diff)
        else:
            raise ValueError('Unknown bootstrap')
        self.bootstrap = bootstrap
        self._pvalues = None
        self._simulated_vals = None
        self._selector = np.ones(self.k, dtype=np.bool)
        self._info = self._display_info()

    def _display_info(self):
        if self.studentize:
            method = 'nested bootstrap' if self.nested else 'asymptotic'
        else:
            method = 'none'
        return OrderedDict(studentization_method=method,
                           bootstrap=str(self.bootstrap),
                           id=hex(id(self)))

    def __str__(self):
        repr = 'SPA('
        for key in self._info:
            if key == 'id':
                continue
            repr += key.replace('_', ' ') + ': ' + self._info[key] + ', '
        repr = repr[:-2] + ')'
        return repr

    def __repr__(self):
        return self.__str__()[:-1] + ', ID:' + self._info['id'] + ')'

    def _repr_html_(self):
        repr = '<strong>SPA</strong>('
        for key in self._info:
            if key == 'id':
                continue
            repr += '<strong>' + key.replace('_', ' ') + '</strong>: ' + self._info[key] + ','
        repr = repr[:-2] + ')'
        return repr

    def reset(self):
        """
        Reset the bootstrap to it's initial state.
        """
        self.bootstrap.reset()
        self._pvalues = None

    def seed(self, value):
        """
        Seeds the bootstrap's random number generator

        Parameters
        ----------
        value : int
            Integer to use as the seed
        """
        self.bootstrap.seed(value)

    def subset(self, selector):
        """
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
        simulated_vals = simulated_vals[:, self._selector, :]
        max_simulated_vals = np.max(simulated_vals, 0)
        loss_diff = self._loss_diff[:, self._selector]

        max_loss_diff = np.max(loss_diff.mean(axis=0))
        pvalues = (max_simulated_vals > max_loss_diff).mean(axis=0)
        self._pvalues = OrderedDict(lower=pvalues[0],
                                    consistent=pvalues[1],
                                    upper=pvalues[2])

    def _simulate_values(self):
        self._compute_variance()
        # 2. Compute invalid columns using criteria for consistent
        self._valid_columns = self._check_column_validity()
        # 3. Compute simulated values
        # Upper always re-centers
        upper_mean = self._loss_diff.mean(0)
        consitent_mean = upper_mean.copy()
        consitent_mean[np.logical_not(self._valid_columns)] = 0.0
        lower_mean = upper_mean.copy()
        # Lower does not re-center those that are worse
        lower_mean[lower_mean < 0] = 0.0
        means = [lower_mean, consitent_mean, upper_mean]
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
            func = lambda x: x.mean(0)
            means = bs.apply(func, reps=self.reps)
            return self.t * means.var(axis=0)
        else:
            t = self.t
            p = 1.0 / self.block_size
            variances = np.sum(demeaned ** 2, 0) / t
            for i in range(1, t):
                kappa = ((1.0 - (i / t)) * ((1 - p) ** i)) + ((i / t) * ((1 - p) ** (t - i)))
                variances += 2 * kappa * np.sum(demeaned[0:(t - i), :] * demeaned[i:t, :], 0) / t
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
        if self._pvalues is None:
            msg = 'compute must be called before pvalues are available.'
            raise RuntimeError(msg)
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
        if self._pvalues is None:
            msg = 'compute must be called before pvalues are available.'
            raise RuntimeError(msg)
        # Subset if neded
        simulated_values = self._simulated_vals[:, self._selector]
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
            String in 'lower', 'consistent', or 'upper' indicating which critical
            value to use.

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
        if pvalue_type not in self._pvalues:
            raise ValueError('Unknown pvalue type')
        if self._pvalues is None:
            msg = 'compute must be called before pvalues are available.'
            raise RuntimeError(msg)
        crit_val = self.critical_values(pvalue=pvalue)[pvalue_type]
        better_models = self._loss_diff.mean(0) > crit_val
        better_models = np.logical_and(better_models, self._selector)
        if isinstance(self.models, pd.DataFrame):
            return list(self.models.columns[better_models])
        else:
            return np.argwhere(better_models).flatten()


class RealityCheck(SPA):
    # Shallow clone of SPA
    pass
