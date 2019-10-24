# TODO Phillips-Ouliaris-Hansen Test For Cointegration
# TODO IntOrder Function
""""
Author: Austin Adams

Cointegration is a time series method used a linear combination of
of multiple time series eliminate common stochastic trends.

"""

from statsmodels.regression.linear_model import OLS
from arch.utility.timeseries import add_trend
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tools.data import _is_using_pandas
from arch.unitroot import ADF, DFGLS, PhillipsPerron
import pandas as pd
import numpy as np


class Cointegration(object):
    """
    Class to test and estimate cointegration coefficents

    Cointegration:
    engle_granger_coef
    dynamic_coefs
    """
    def engle_granger_coef(self, y0, y1, trend='c',
                           max_lags=None, autolag='AIC',
                           method='ADF',
                           normalize=True, check=True):
        """
        Engle-Granger Cointegration Coefficient Calculations.

        This equation takes a linear combination of two L(1) time series to
        create a L(0) or stationary time series.

        This is useful if the two series have a similar stochastic long-term
        trend, as it eliminates them and allows you

        Parameters
        ----------
        y0 : array_like
            The first element in cointegrated system. Must be 1-d.
        y1 : array_like
            The remaining elements in cointegrated system.
        trend : str {'c', 'ct'}
            The trend term included in regression for cointegrating equation.

            * 'c' : constant.
            * 'ct' : constant and linear trend.
            * also available quadratic trend 'ctt', and no constant 'nc'.
        maxlag : None or int
            Argument for `adfuller`, largest or given number of lags.
        method: string
            Determines what method is used to check for a stationary process
            The available are DFGLS, PhillipsPerron, ADF.
        autolag : str
            * If 'AIC' (default) or 'BIC', then the number of lags is chosen
            to minimize the corresponding information criterion.
        normalize: boolean, optional
            As there are infinite scalar combinations that will produce the
            factor, this normalizes the first entry to be 1.
        check: boolean, optional
            Checks if the series has a possible cointegration factor using the
            Engle-Granger Cointegration Test

        Returns
        -------
        coefs: array
            A vector that will create a L(0) time series if a combination
            exists.

        Notes
        -----
        The series should be checked independently for their integration
        order. The series must be L(1) to get consistent results. You can
        check this by using the int_order function.

        References
        ----------
        .. [1] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution
        Functions for Unit-Root and Cointegration Tests." Journal of
        Business & Economics Statistics, 12.2, 167-76.
        .. [2] MacKinnon, J.G. 2010.  "Critical Values for Cointegration
        Tests." Queen's University, Dept of Economics Working Papers 1227.
        http://ideas.repec.org/p/qed/wpaper/1227.html
        .. [3] Hamilton, J. D. (1994). Time series analysis
        (Vol. 2, pp. 690-696). Princeton, NJ: Princeton university press.
        """
        nobs, k_vars = y1.shape

        y1 = add_trend(y1, trend=trend, prepend=False)

        eg_model = OLS(y0, y1).fit()
        coefs = eg_model.params[0: k_vars]

        if check:
            if method == 'DFGLS':
                unit_root = DFGLS(
                                eg_model.resid,
                                trend=trend,
                                max_lags=max_lags,
                                method=method)
            elif method == 'PhillipsPerron':
                unit_root = PhillipsPerron(eg_model.resid, trend=trend)
            else:
                unit_root = ADF(eg_model.resid,
                                trend=trend,
                                max_lags=max_lags,
                                method=method)

            if unit_root.pvalue > .1:
                print('The null hypothesis cannot be rejected')

        if normalize:
            coefs = coefs / coefs[0]

        return coefs

    def dynamic_coefs(
                    self, y0, y1, n_lags=None,
                    trend='c', auto_lag='bic',
                    normalize=True, reverse=False):
        """
        Dynamic Cointegration Coefficient Calculation.

        This equation takes a linear combination of multiple L(1) time
        series to create a L(0) or stationary time series.

        This is useful if the two series have a similar stochastic long-term
        trend, as it eliminates them and allows you.

        Unlike Engle-Granger, this method uses dynamic regression - taking
        an equal combination of lags and leads of the differences of the
        series - to create a more accurate parameter vector. This method
        calculates the lag-lead matricies for the given lag values or searches
        for the best amount of lags using BIC calculations. Once the optimal
        value is found, the calculation is done and returned. The optimal
        lag can be found by using dot notation and finding max_lag. You
        can also find the model by using .model.

        Parameters
        ----------
        y0 : array_like
            The first element in cointegrated system. Must be 1-d.
        y1 : array_like
            The remaining elements in cointegrated system.
        n_lags: int, array, None
            This determines which values the function should search for the
            best vector.

            * int: If an int, the calculation is done for only that lag
            * array: If an array of two integers, the first value is where
                        the search begins and the second is where it ends
            * None: If None is given, the function searches from 2 to
                        ceiling of the cube root of the number of observations
                        divided by two plus two in order to ensure at least
                        one value is searched.
                        I.E last_lag = (n_obs**(1/3) / 2) + 2

        trend : str {'c', 'ct'}
            The trend term included in regression for cointegrating equation.

            * 'c' : constant.
            * 'ct' : constant and linear trend.
            * also available quadratic trend 'ctt', and no constant 'nc'.

        normalize: Boolean
            If true, the first entry in the parameter vector is normalized to
            one and everything else is divided by the first entry. This is
            because any cointegrating vector could be multiplied by a scalar
            and still be a cointegrating vector.
        reverse: Boolean
            The series must be ordered from the latest data points to the last.
            This is in order to calculate the differences. Using this, you can
            reverse the ordering of your data points.

        Returns
        -------
        coefs: array
            A vector that will create a L(0) time series if a
            combination sexists.

        Notes
        -----
        The data must go from the latest observations to the earliest. If not,
        the coef vector will be the opposite sign.

        The series should be checked independently for their integration order.
        The series must be L(1) to get consistent results. You can check this
        by using the int_order function.

        References
        ----------
        .. [1] Stock, J. H., & Watson, M. W. (1993). A simple estimator of
        cointegrating vectors in higher order integrated systems.
        Econometrica: Journal of the Econometric Society, 783-820.
        .. [2] Hamilton, J. D. (1994). Time series analysis
        (Vol. 2, pp. 690-696). Princeton, NJ: Princeton university press.
        """
        self.ic = []
        self.max_val = []
        self.model = ''
        self.coefs = []

        y1 = add_trend(y1, trend=trend, prepend=True)
        y1 = y1.reset_index(drop=True)
        if reverse:
            y0, y1 = y0[::-1], y1[::-1]

        if _is_using_pandas(y0, y1):
            columns = list(y1.columns)

        else:
            # Need to check if NumPy, because I can only support those two
            n_obs, k = y1.shape
            columns = [f'Var_{x}' for x in range(k)]
            y0, y1 = pd.DataFrame(y0), pd.DataFrame(y1)

        # If none or interval, search for it using BIC or AIC
        if n_lags is None or len(n_lags) == 2:
            if len(n_lags) == 2:
                start, end = int(n_lags[0]), int(n_lags[1]) + 1
            elif n_lags is None:
                start = 2
                end = int(np.ceil(n_obs ** (1 / 3) / 2) + 2)

            n_obs, k = y1.shape
            dta = pd.DataFrame(np.diff(a=y1, n=1, axis=0))
            for lag in range(start, end):

                df1 = pd.DataFrame(lagmat(dta, lag + 1, trim='backward'))
                cols = dict(zip(list(df1.columns)[::-1][0:k][::-1], columns))
                df1 = df1.rename(columns=cols)

                df2 = pd.DataFrame(lagmat(dta, lag, trim='forward'))

                lags_leads = pd.concat([df1, df2], axis=1, join='outer')
                lags_leads = lags_leads.drop(list(range(0, lag)))
                lags_leads = lags_leads.reset_index(drop=True)

                lags_leads = lags_leads.drop(
                    list(range(len(lags_leads) - lag, len(lags_leads))))

                lags_leads = lags_leads.reset_index(drop=True)
                data_y = y0.drop(list(range(0, lag))).reset_index(drop=True)
                data_y = data_y.drop(
                    list(range(len(data_y) - lag - 1, len(data_y))))
                data_y = data_y.reset_index(drop=True)

                if auto_lag == 'aic':
                    self.ic.append([OLS(data_y, lags_leads).fit().aic, lag])
                elif auto_lag == 'bic':
                    self.ic.append([OLS(data_y, lags_leads).fit().bic, lag])

            self.max_val = max(self.ic, key=lambda item: item[0])
            self.max_val = self.max_val[1]

        elif len(n_lags) == 1:
            self.max_val = int(n_lags)

        else:
            raise('Make sure your lags are in one of the required forms.')

        dta = pd.DataFrame(np.diff(a=y1, n=1, axis=0))
        # Create a matrix of the lags, this also retains the original matrix,
        # which is why max_val + 1
        df1 = pd.DataFrame(lagmat(dta, self.max_val + 1, trim='backward'))

        # Rename the columns, as we need to keep track of them. We know the
        # original will be the final values
        cols = dict(zip(list(df1.columns)[::-1][0:k][::-1], columns))
        df1 = df1.rename(columns=cols)

        # Do the same, but these are leads, this does not keep the
        # original matrix, thus max_val
        df2 = pd.DataFrame(lagmat(dta, self.max_val, trim='forward'))

        # There are missing data due to the lags and leads, we concat
        # the frames and drop the values of which are missing.
        lags_leads = pd.concat([df1, df2], axis=1, join='outer')
        lags_leads = lags_leads.drop(list(range(0, self.max_val)))
        lags_leads = lags_leads.reset_index(drop=True)
        lags_leads = lags_leads.drop(
            list(range(len(lags_leads) - self.max_val, len(lags_leads))))
        lags_leads.reset_index(drop=True)

        # We also need to do this for the endog values, we need to
        # drop 1 extra due to a loss from first differencing.
        # This will be at the end of the matrix.
        data_y = y0.drop(list(range(0, self.max_val))).reset_index(drop=True)
        data_y = data_y.drop(list(range(len(data_y) - self.max_val -
                                        1, len(data_y))))
        data_y = data_y.reset_index(drop=True)

        self.model = OLS(data_y, lags_leads).fit()

        self.coefs = self.model.params[list(y1.columns)]

        if normalize:
            self.coefs = self.coefs / self.coefs[0]

        return(self.coefs)
