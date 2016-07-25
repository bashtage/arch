from numpy import empty, reshape, log, pi
from numpy.linalg import slogdet, inv
from numpy.random import standard_normal

from ..univariate.distribution import Distribution


class Normal(Distribution):
    """
    Multivariate standard normal distribution for use with ARCH models
    """

    def __init__(self, k):
        super(Normal, self).__init__('Normal')
        self.name = str(k) + '-dimensional Multivariate Normal'
        self.k = k

    def constraints(self):
        return empty(0), empty(0)

    def bounds(self, resids):
        return tuple([])

    def loglikelihoood(self, parameters, resids, sigma2, individual=False):
        r"""
        Computes the log-likelihood of assuming residuals follow a
        multivariate normal distribution, conditional on the variance

        Parameters
        ----------
        parameters : empty array
            The normal likelihood has no shape parameters
        resids  : 2-d array, float
            The residuals to use in the log-likelihood calculation (nobs by k)
        sigma2 : 3-d array, float
            Conditional covariances of resids (nobs by k by k)
        individual : bool, optional
            Flag indicating whether to return the vector of individual log
            likelihoods (True) or the sum (False)

        Returns
        -------
        ll : float64
            The log-likelihood

        Notes
        -----
        The log-likelihood of a single data point x is

        .. math::

            \ln f\left(x\right)=-\frac{1}{2}\left(k\ln\left(2\pi\right)
            -\ln\left|\Sigma\right|-r^{\prime}\Sigma^{-1}r\right)

        """
        t, k = resids.shape
        lls = empty(t)
        llf_const = k * log(2 * pi)
        for i in range(t):
            _, logdet = slogdet(sigma2[i])
            lls[i] = -0.5 * (llf_const + logdet +
                             resids[i].dot(inv(sigma2[i])).dot(resids[i].T))
        if individual:
            return lls
        else:
            return sum(lls)

    def starting_values(self, std_resid):
        """
        Parameters
        ----------
        std_resid : 1-d array
            Estimated standardized residuals to use in computing starting
            values for the shape parameter

        Returns
        -------
        sv : empty array
            The normal distribution has no shape parameters

        """
        return empty(0)

    def _simulator(self, random_state):
        if random_state is None:
            rng = standard_normal
        else:
            rng = random_state.standard_normal

        def _sim(nobs):
            return reshape(rng(nobs * self.k), (nobs, self.k))

        return _sim

    def simulate(self, parameters, random_state=None):
        return self._simulator(random_state=random_state)

    def parameter_names(self):
        return []
