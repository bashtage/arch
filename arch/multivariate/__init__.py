"""
Multivariate ARCH models.

Provides CCC-GARCH (Bollerslev 1990) and DCC-GARCH (Engle 2002) for
modelling time-varying covariance matrices of multivariate return series.
"""

from arch.multivariate.dcc import CCCModel, DCCModel, CCCResult, DCCResult, simulate_dcc

__all__ = [
    "CCCModel",
    "DCCModel",
    "CCCResult",
    "DCCResult",
    "simulate_dcc",
]
