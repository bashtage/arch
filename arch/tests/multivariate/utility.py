from collections import namedtuple
from itertools import product

import numpy as np
import pandas as pd
import pytest

from arch.multivariate.distribution import MultivariateNormal
from arch.multivariate.mean import ConstantMean, VARX
from arch.multivariate.volatility import ConstantCovariance, EWMACovariance

dataset = namedtuple('dataset', ['y', 'x', 'lags', 'constant'])


def generate_data(nvar, nobs, nexog, common, lags, constant, pandas):
    rs = np.random.RandomState(1)
    y = rs.standard_normal((nobs, nvar))
    if nexog > 0:
        x = rs.standard_normal((nobs, nexog))
    else:
        x = None
    if pandas:
        index = pd.date_range('2000-1-1', periods=nobs)
        cols = ['y{0}'.format(i) for i in range(nvar)]
        y = pd.DataFrame(y, index=index, columns=cols)
        if nexog > 0:
            cols = ['x{0}'.format(i) for i in range(nexog)]
            x = pd.DataFrame(x, index=index, columns=cols)
    if not common:
        if pandas:
            x = {c: x for c in y}
        else:
            x = [x] * nvar
    return dataset(y=y, x=x, lags=lags, constant=constant)


