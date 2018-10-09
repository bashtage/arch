import numpy as np
import pandas as pd


class TimeSeries(object):
    def __init__(self, y, name='y', nvar=None):
        self._y_original = y
        if y is None:
            self._y_arr = np.empty((0, nvar))
        else:
            self._y_arr = np.asarray(y)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            self._pandas_input = True
            y_pd = y
            self._time_index = isinstance(y_pd.index, pd.DatetimeIndex)
        else:
            self._pandas_input = False
            self._time_index = False
            if self._y_arr.ndim == 1:
                y_pd = pd.Series(self._y_arr, name=name)
            else:
                cols = [name + '.{0}'.format(i) for i in range(self._y_arr.shape[1])]
                y_pd = pd.DataFrame(self._y_arr, columns=cols)
        self._y_pd = y_pd

    @property
    def array(self):
        return self._y_arr

    @property
    def frame(self):
        return self._y_pd

    @property
    def shape(self):
        return self._y_arr.shape

    @property
    def original(self):
        return self._y_original

    def index_loc(self, idx, default):
        if idx is None:
            return default
        try:
            return self._y_pd.index.get_loc(idx)
        except KeyError:
            if isinstance(idx, int):
                return idx
        raise KeyError('idx is not in index and is not an integer')

    @property
    def pandas_input(self):
        return self._pandas_input