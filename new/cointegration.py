#TODO Phillips-Ouliaris-Hansen Test For Cointegration

""""
Author: Austin Adams

Cointegration is a time series method used a linear combination of 
of multiple time series eliminate common stochastic trends. 

"""


from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import add_trend, lagmat
from statsmodels.tools.validation import string_like, array_like
from statsmodels.tools.data import _is_using_pandas
import pandas as pd
import numpy as np

class Cointegration(object):
    """
    Class to test and estimate cointegration coefficents

    Parameters
    ----------
    No idea
    ----------
    """
    def engle_granger_test(self, y0, y1, trend = 'c', 
                    method = 'aeg', maxlag = None, 
                    autolag='aic'):
        """
        Test for no-cointegration of a univariate equation.

        The null hypothesis is no cointegration. Variables in y0 and y1 are
        assumed to be integrated of order 1, I(1).

        This uses the augmented Engle-Granger two-step cointegration test.
        Constant or trend is included in 1st stage regression, i.e. in
        cointegrating equation.

        **Warning:** The autolag default has changed compared to statsmodels 0.8.
        In 0.8 autolag was always None, no the keyword is used and defaults to
        'aic'. Use `autolag=None` to avoid the lag search.

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

        method : {'aeg'}
            Only 'aeg' (augmented Engle-Granger) is available.
        maxlag : None or int
            Argument for `adfuller`, largest or given number of lags.
        autolag : str
            Argument for `adfuller`, lag selection criterion.

            * If None, then maxlag lags are used without lag search.
            * If 'AIC' (default) or 'BIC', then the number of lags is chosen
            to minimize the corresponding information criterion.
            * 't-stat' based choice of maxlag.  Starts with maxlag and drops a
            lag until the t-statistic on the last lag length is significant
            using a 5%-sized test.
        return_results : bool
            For future compatibility, currently only tuple available.
            If True, then a results instance is returned. Otherwise, a tuple
            with the test outcome is returned. Set `return_results=False` to
            avoid future changes in return.

        Returns
        -------
        coint_t : float
            The t-statistic of unit-root test on residuals.
        pvalue : float
            MacKinnon's approximate, asymptotic p-value based on MacKinnon (1994).
        crit_value : dict
            Critical values for the test statistic at the 1 %, 5 %, and 10 %
            levels based on regression curve. This depends on the number of
            observations.

        Notes
        -----
        The Null hypothesis is that there is no cointegration, the alternative
        hypothesis is that there is cointegrating relationship. If the pvalue is
        small, below a critical size, then we can reject the hypothesis that there
        is no cointegrating relationship.

        P-values and critical values are obtained through regression surface
        approximation from MacKinnon 1994 and 2010.

        If the two series are almost perfectly collinear, then computing the
        test is numerically unstable. However, the two series will be cointegrated
        under the maintained assumption that they are integrated. In this case
        the t-statistic will be set to -inf and the pvalue to zero.

        TODO: We could handle gaps in data by dropping rows with nans in the
        Auxiliary regressions. Not implemented yet, currently assumes no nans
        and no gaps in time series.

        References
        ----------
        .. [1] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions
        for Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.
        .. [2] MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
        Queen's University, Dept of Economics Working Papers 1227.
        http://ideas.repec.org/p/qed/wpaper/1227.html
        """
        coint_t, pvalue, crit_value = coint(y0, y1, trend, method, maxlag, autolag)
        
        return coint_t, pvalue, crit_value
    
    def engle_granger_coef(self, y0, y1, trend='c',
                    method='aeg', maxlag=None,
                    autolag='aic', normalize = True, debug = True):
        """
        Engle-Granger Cointegration Coefficient.

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

        method : {'aeg'}
            Only 'aeg' (augmented Engle-Granger) is available.
        maxlag : None or int
            Argument for `adfuller`, largest or given number of lags.
        autolag : str
            Argument for `adfuller`, lag selection criterion.

            * If None, then maxlag lags are used without lag search.
            * If 'AIC' (default) or 'BIC', then the number of lags is chosen
            to minimize the corresponding information criterion.
            * 't-stat' based choice of maxlag.  Starts with maxlag and drops a
            lag until the t-statistic on the last lag length is significant
            using a 5%-sized test.
        normalize: boolean, optional
            As there are infinite scalar combinations that will produce the
            factor, this normalizes the first entry to be 1.
        debug: boolean, optional
            Checks if the series has a possible cointegration factor using the
            Engle-Granger Cointegration Test
        
        Returns
        -------
        coefs: array
            A vector that will create a L(0) time series if a combination exists.

        Notes
        -----
        The series should be checked independently for their integration order.
        The series must be L(1) to get consistent results. 

        References
        ----------
        .. [1] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions
        for Unit-Root and Cointegration Tests." Journal of Business & Economics
        Statistics, 12.2, 167-76.
        .. [2] MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
        Queen's University, Dept of Economics Working Papers 1227.
        http://ideas.repec.org/p/qed/wpaper/1227.html
        .. [3] Hamilton, J. D. (1994). Time series analysis (Vol. 2, pp. 690-696). 
        Princeton, NJ: Princeton university press.
        """
        if debug == True:
            coint_t, pvalue, crit_value = coint(y0, y1, trend, method, maxlag, autolag)
            if pvalue >= .10:
                print(f'The null hypothesis of no cointegration cannot be rejected')

        trend = string_like(trend, 'trend', options = ('c', 'nc', 'ct', 'ctt'))
        nobs, k_vars = y1.shape

        y1 = add_trend(y1, trend = trend, prepend = False)

        eg_model = OLS(y0, y1).fit()
        coefs = eg_model.params[0: k_vars]
        
        if normalize == True:
            coefs = coefs / coefs[0]
        
        return coefs

    def dynamic_coeffs(self, y0, y1, n_lags=None, trend='c',
                        normalize = True, reverse = False):
        # y0 is endog
        # y1 is exog
        self.bics = []
        self.max_val = []
        self.model = ''
        self.params = []

        trend = string_like(trend, 'trend', options = ('c', 'nc', 'ct', 'ctt'))
        y1 = add_trend(y1, trend = trend, prepend = True)
        y1 = y1.reset_index(drop = True)
        if reverse == True:
            y0, y1 = y0[::-1], y1[::-1]

        if _is_using_pandas(y0, y1):
            columns = list(y1.columns)

        else:
            # Need to check if NumPy, because I can only support those two
            n_obs, k = y1.shape
            columns = [f'Var_{x}' for x in range(k)]
            y0, y1 = pd.DataFrame(y0), pd.DataFrame(y1)

        if n_lags is None:
            n_obs, k = y1.shape
            dta = pd.DataFrame(np.diff(a = y1, n = 1, axis = 0))
            for lag in range(2, int(np.ceil(n_obs ** (1/3) / 2) + 2)):
                
                df1 = pd.DataFrame(lagmat(dta, lag+1 , trim = 'backward'))
                cols = dict(zip(list(df1.columns)[::-1][0:k][::-1], columns))
                df1 = df1.rename(columns = cols)
                
                df2 = pd.DataFrame(lagmat(dta, lag , trim = 'forward'))

                lags_leads = pd.concat([df1, df2], axis = 1, join = 'outer')
                lags_leads = lags_leads.drop(list(range(0, lag))).reset_index(drop = True)
                lags_leads = lags_leads.drop(list(range(len(lags_leads) - lag, len(lags_leads)))).reset_index(drop = True)
                
                data_y = y0.drop(list(range(0, lag))).reset_index(drop = True)
                data_y = data_y.drop(list(range(len(data_y) - lag - 1, len(data_y)))).reset_index(drop = True)

                self.bics.append([OLS(data_y, lags_leads).fit().bic, lag])

            self.max_val = max(self.bics, key=lambda item: item[0])
            self.max_val = self.max_val[1]

        elif len(n_lags) == 2:
            start, end = int(n_lags[0]), int(n_lags[1])
            n_obs, k = y1.shape
            dta = pd.DataFrame(np.diff(a = y1, n = 1, axis = 0))

            for lag in range(start, end+1):
                df1 = pd.DataFrame(lagmat(dta, lag+1 , trim = 'backward'))
                cols = dict(zip(list(df1.columns)[::-1][0:k][::-1], columns))
                df1 = df1.rename(columns = cols)
                
                df2 = pd.DataFrame(lagmat(dta, lag , trim = 'forward'))

                lags_leads = pd.concat([df1, df2], axis = 1, join = 'outer')
                lags_leads = lags_leads.drop(list(range(0, lag))).reset_index(drop = True)
                lags_leads = lags_leads.drop(list(range(len(lags_leads) - lag, len(lags_leads)))).reset_index(drop = True)
                
                data_y = y0.drop(list(range(0, lag))).reset_index(drop = True)
                data_y = data_y.drop(list(range(len(data_y) - lag - 1, len(data_y)))).reset_index(drop = True)

                self.bics.append([OLS(data_y, lags_leads).fit().bic, lag])

            self.max_val = max(self.bics, key=lambda item: item[0])
            self.max_val = self.max_val[1]
        
        elif len(n_lags) == 1:
            self.max_val = int(n_lags)

        else:
            raise('Please make sure your provided lags are in one of the three required forms.')

        dta = pd.DataFrame(np.diff(a = y1, n = 1, axis = 0))
        # Create a matrix of the lags, this also retains the original matrix, which is why max_val + 1
        df1 = pd.DataFrame(lagmat(dta, self.max_val+1 , trim = 'backward'))

        # Rename the columns, as we need to keep track of them. We know the original will be the final values
        cols = dict(zip(list(df1.columns)[::-1][0:k][::-1], columns))
        df1 = df1.rename(columns = cols)

        # Do the same, but these are leads, this does not keep the original matrix, thus max_val
        df2 = pd.DataFrame(lagmat(dta, self.max_val , trim = 'forward'))

        # There are missing data due to the lags and leads, we concat the frames and drop the values of which are missing.
        lags_leads = pd.concat([df1, df2], axis = 1, join = 'outer')
        lags_leads = lags_leads.drop(list(range(0, self.max_val))).reset_index(drop = True)
        lags_leads = lags_leads.drop(list(range(len(lags_leads) - self.max_val, len(lags_leads)))).reset_index(drop = True)

        # We also need to do this for the endog values, we need to drop 1 extra due to a loss from first differencing.
        # This will be at the end of the matrix.
        data_y = y0.drop(list(range(0, self.max_val))).reset_index(drop = True)
        data_y = data_y.drop(list(range(len(data_y) - self.max_val - 1, len(data_y)))).reset_index(drop = True)

        self.model = OLS(data_y, lags_leads).fit()

        self.params = self.model.params[list(y1.columns)]
        
        if normalize == True:
            self.params = self.params / self.params[0]

        return(self.params)




        
    

