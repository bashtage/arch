""""
Author: Austin Adams

Cointegration is a time series method used a linear combination of 
of multiple time series eliminate common stochastic trends. 

"""


from statsmodels.tsa.stattools import coint
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
                    autolag='aic'):
        
