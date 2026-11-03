"""
Contains values used to approximate the critical value and p-value from DFGLS
statistics

These have been computed using the methodology of MacKinnon (1994) and (2010)
simulation. See dfgls_critival_values_simulation for implementation.

Trend c is based on 2000 experiments of 250,000 simulations.
Trend ct is based on 2000 experiments of 250,000 simulations.
"""

from numpy import array, inf

dfgls_cv_approx = {
    "c": array(
        [
            [-2.56781264e00, -2.05642164e01, 1.81706248e02, -1.74925345e03],
            [-1.94372334e00, -2.17178566e01, 2.60868105e02, -2.27376815e03],
            [-1.62009027e00, -2.32676246e01, 3.06807184e02, -2.58327340e03],
        ]
    ),
    "ct": array(
        [
            [-3.40687633, -21.70376107, 27.73389824, -826.67043393],
            [-2.84688238, -19.65710453, 83.05317014, -779.33088058],
            [-2.55897456, -19.4004776, 115.20169869, -825.34125409],
        ]
    ),
}

dfgls_tau_max = {"c": inf, "ct": 10.29976583}

dfgls_tau_min = {"c": -20.21603375, "ct": -16.19373563}

dfgls_tau_star = {"c": -0.90809, "ct": -2.66464}

dfgls_large_p = {
    "c": [0.5000589434, 0.9429346486, -0.07798850876, 0.03725033442],
    "ct": [2.595735355, 1.684621774, 0.04656658257, -0.00830738613],
}

dfgls_small_p = {
    "c": [0.6452877771, 1.232072596, 0.03047265875],
    "ct": [2.281254794, 1.50756695, 0.04654784368],
}
