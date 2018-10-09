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


NVAR = [1, 2, 10]
NOBS = [100, 500]
NEXOG = [0, 1, 3]
COMMON = [True, False]
LAGS = [1, [1], 3, [1, 3], None]
CONSTANT = [True, False]
PANDAS = [True, False]
params = list(product(NVAR, NOBS, NEXOG, COMMON, LAGS, CONSTANT, PANDAS))
ID = """nvar={0}, nobs={1}, nexog={2}, common={3}, lags={4}, constant={5}, pandas={6}"""
ids = [ID.format(*map(str, p)) for p in params]

#@pytest.fixture(scope='module', params=params, ids=ids)
#def var_data(request):
#    nvar, nobs, nexog, common, lags, constant, pandas = request.param
#    return generate_data(nvar, nobs, nexog, common, lags, constant, pandas)

for id, param in zip(ids,params):
    nvar, nobs, nexog, common, lags, constant, pandas = param
    print(id)

    var_data = generate_data(nvar, nobs, nexog, common, lags, constant, pandas)
    vol = ConstantCovariance()
    dist = MultivariateNormal()
    lags = var_data.lags
    constant = var_data.constant
    y = var_data.y
    x = var_data.x
    mod = VARX(y, x, lags=lags, constant=constant, volatility=vol, distribution=dist)
    mod.fit(cov_type='mle')
