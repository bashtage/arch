from numpy import empty

from .distribution import Normal
from .volatility import ConstantCovariance
from ..common.model import ARCHModel
from ..utility.array import ensure2d


class MultivariateARCHModel(ARCHModel):
    """
    Abstract base class for multivariate mean models in ARCH processes.
    Specifies the conditional mean process.

    All public methods that raise NotImplementedError should be overridden by
    any subclass.  Private methods that raise NotImplementedError are optional
    to override but recommended where applicable.
    """

    def __init__(self, y=None, volatility=None, distribution=None,
                 hold_back=None, last_obs=None):

        if y is not None:
            y = ensure2d(y, 'y')
        else:
            y = ensure2d(empty((0, 0)), 'y')

        super(MultivariateARCHModel, self).__init__(y, volatility,
                                                    distribution, hold_back,
                                                    last_obs)

        if self.volatility is None:
            self.volatility = ConstantCovariance()

        if distribution is None:
            self.distribution = Normal()
