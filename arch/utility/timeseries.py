import numpy as np
import pandas as pd


class ColumnNameConflict(Warning):
    pass


column_name_conflict_doc = """
Some of the column named being added were not unique and have been renamed.

             {0}
"""


def _enforce_unique_col_name(existing, new):
    converted_names = []
    unique_names = list(new[:])
    for i, n in enumerate(new):
        if n in existing:
            original_name = n
            fixed_name = n
            duplicate_count = 0
            while fixed_name in existing:
                fixed_name = n + '_' + str(duplicate_count)
                duplicate_count += 1
            unique_names[i] = fixed_name
            converted_names.append(
                '{0}   ->   {1}'.format(original_name, fixed_name))
    if converted_names:
        import warnings

        ws = column_name_conflict_doc.format('\n    '.join(converted_names))
        warnings.warn(ws, ColumnNameConflict)

    return unique_names


def add_trend(x=None, trend="c", prepend=False, nobs=None,
              has_constant='skip'):
    """
    Adds a trend and/or constant to an array.

    Parameters
    ----------
    x : array-like or None
        Original array of data. If None, then nobs must be a positive integer
    trend : str {"c","t","ct","ctt"}
        "c" add constant only
        "t" add trend only
        "ct" add constant and linear trend
        "ctt" add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of x.
    n : int, positive
        Positive integer containing the length of the trend series.  Only used
        if x is none.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant already
        exists in X. 'raise' will raise an error. 'add' will duplicate a
        constant. 'skip' will return the data without change. 'skip' is the
        default.

    Notes
    -----
    Returns columns as ["ctt","ct","t","c"] whenever applicable. There is
    currently no checking for an existing trend.
    """
    trend = trend.lower()
    if trend == 'c':
        trend_order = 0
    elif trend == 'ct' or trend == 't':
        trend_order = 1
    elif trend == 'ctt':
        trend_order = 2
    else:
        raise ValueError('trend %s not understood' % trend)
    if x is not None:
        nobs = len(np.asanyarray(x))
    elif nobs is None or nobs <= 0:
        raise ValueError('nobs must be a positive integer if x is None')
    trend_array = np.vander(np.arange(1, nobs + 1, dtype=np.float64),
                            trend_order + 1)
    # put in order ctt
    trend_array = np.fliplr(trend_array)
    if trend == 't':
        trend_array = trend_array[:, 1:]
        # check for constant
    if x is None:
        return trend_array
    x_array = np.asarray(x)
    if 'c' in trend and \
            np.any(np.logical_and(np.ptp(x_array, axis=0) == 0,
                                  np.all(x_array != 0, axis=0))):
        if has_constant == 'raise':
            raise ValueError('x already contains a constant')
        elif has_constant == 'add':
            pass
        elif has_constant == 'skip' and trend in ('c', 'ct', 'ctt'):
            trend_array = trend_array[:, 1:]
    if isinstance(x, pd.DataFrame):
        columns = ('const', 'trend', 'quadratic_trend')
        if trend == 't':
            columns = (columns[1],)
        else:
            columns = columns[0:trend_order + 1]
        columns = _enforce_unique_col_name(x.columns, columns)
        trend_array = pd.DataFrame(trend_array, index=x.index, columns=columns)
        if prepend:
            x = trend_array.join(x)
        else:
            x = x.join(trend_array)
    else:
        if prepend:
            x = np.column_stack((trend_array, x))
        else:
            x = np.column_stack((x, trend_array))

    return x
