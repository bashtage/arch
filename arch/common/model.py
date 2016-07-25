import numpy as np
import pandas as pd

from ..compat.python import add_metaclass
from ..utility.array import DocStringInheritor, ensure1d, ensure2d


@add_metaclass(DocStringInheritor)
class ARCHModel(object):
    """
    Abstract base class for mean models with ARCH processes.
    Specifies the conditional mean process.

    All public methods that raise NotImplementedError should be overridden by
    any subclass.  Private methods that raise NotImplementedError are optional
    to override but recommended where applicable.
    """

    def __init__(self, y=None, volatility=None, distribution=None, hold_back=None):

        # Set on model fit
        self._fit_indices = None
        self._fit_y = None

        # Set on model fit
        self._fit_indices = None
        self._fit_y = None

        self._is_pandas = isinstance(y, (pd.DataFrame, pd.Series))
        if y is not None:
            ndim = y.ndim
            if ndim == 1:
                self._y_pd = ensure1d(y, 'y', series=True)
            else:
                self._y_pd = ensure2d(y, 'y', dataframe=True)
        else:
            self._y_pd = ensure1d(np.empty((0,)), 'y', series=True)

        self._y = np.asarray(self._y_pd)
        self._y_original = y

        self.hold_back = hold_back
        self._hold_back = 0 if hold_back is None else hold_back

        self._volatility = volatility
        self._distribution = distribution
        self._backcast = None
        self._var_bounds = None
