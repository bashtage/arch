#TODO Phillips-Ouliaris-Hansen Test For Cointegration

""""
Author: Austin Adams

Cointegration is a time series method used a linear combination of 
of multiple time series eliminate common stochastic trends. 

"""


from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import add_trend
from statsmodels.tools.validation import string_like, array_like
class Cointegration(object):
    """
    Class to test and estimate cointegration coefficents

    Parameters
    ----------
    No idea
    ----------
    """
    def __init__():
        """
        Not sure yet
        """

    def engle_granger_test(y0, y1, trend = 'c', 
                    method = 'aeg', maxlag = None, 
                    autolag='aic'):
        """
        This is utilized from StatsModels
        """
        coint_t, pvalue, crit_value = coint(y0, y1, trend, method, maxlag, autolag)
        
        return coint_t, pvalue, crit_value
    
    def engle_granger_coef(y0, y1, trend='c',
                    method='aeg', maxlag=None,
                    autolag='aic', check = True, normalize = True):
        
        coint_t, pvalue, crit_value = coint(y0, y1, trend, method, maxlag, autolag)
        
        if pvalue >= .10 and debug == True:
            print(f'The null hypothesis of no cointegration cannot be rejected')

        
        trend = string_like(trend, 'trend', options = ('c', 'nc', 'ct', 'ctt'))
        nobs, k_vars = y1.shape

        y1 = add_trend(y1, trend = trend, prepend = False)

        eg_model = OLS(y0, y1).fit()
        coefs = eg_model.params[0: k_vars]
        
        if normalize == True:
            coefs = coefs / coefs[0]
        
        return coefs

    def dynamic_coeffs(y0, y1, trend, n_lags, trend='c',
                    method='aeg', maxlag=None,
                    autolag='aic', check = True, normalize = True):
        
``


        
    

