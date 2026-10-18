"""
Long-run covariances from R's sandwich package for PreWhitenedRecolored.

x is the demeaned first difference of the AAA and BAA bond yields in
arch.data.default::

    import arch.data.default as default

    x = default.load().diff().dropna()
    x = x - x.mean()

The values are the long-run covariances of the VAR(order) prewhitened
estimators computed with R 4.6.1 and sandwich 3.1.3 using the calls in
``R_CALLS``, where ``mod`` has class ``fakemod`` and ``estfun`` returns the
matrix x, ``P`` is the order and ``A`` is ``adjust``. The prewhitening VAR in R
has no intercept so x must be demeaned and arch must use ``center=False``.
R's ``adjust=TRUE`` multiplies the estimate by n / (n - k), where k is the
number of columns of x, so it corresponds to ``df_adjust=2`` here. The
bandwidth in R is the bandwidth of arch plus 1 for the Bartlett and Parzen
kernels, and the same as the bandwidth in arch for the Quadratic Spectral
kernel. R computes its own kernel weights. The VAR-HAC estimator
(``kernel=None``) is the single weight 1.

The dict is keyed by (adjust, order, kernel, bandwidth) and its values are the
2 by 2 long-run covariance.
"""

R_CALLS = {
    "bartlett": (
        "kernHAC(mod, prewhite=P, kernel='Bartlett', bw=7, adjust=A, sandwich=FALSE)"
    ),
    "parzen": (
        "kernHAC(mod, prewhite=P, kernel='Parzen', bw=9, adjust=A, sandwich=FALSE)"
    ),
    "quadratic-spectral": (
        "kernHAC(mod, prewhite=P, kernel='Quadratic Spectral', bw=5, adjust=A, "
        "sandwich=FALSE)"
    ),
    None: "meatHAC(mod, prewhite=P, weights=1, adjust=A)",
}

SANDWICH_LONG_RUN = {
    (False, 1, "bartlett", 6.0): [
        [0.05050759297838545, 0.05266819942559444],
        [0.05266819942559445, 0.07824014446147748],
    ],
    (False, 1, "parzen", 8.0): [
        [0.04789940143619814, 0.050402707870855226],
        [0.050402707870855226, 0.07688108396478027],
    ],
    (False, 1, "quadratic-spectral", 5.0): [
        [0.046657620664458735, 0.04868484426263177],
        [0.04868484426263176, 0.07318182129793134],
    ],
    (False, 1, None, None): [
        [0.059814313943717605, 0.05871335206484996],
        [0.05871335206484996, 0.09143612024834191],
    ],
    (False, 2, "bartlett", 6.0): [
        [0.04317200894701753, 0.04721342438703278],
        [0.04721342438703278, 0.07280126018283836],
    ],
    (False, 2, "parzen", 8.0): [
        [0.04225042991052909, 0.04611017912551866],
        [0.04611017912551865, 0.07224410810479824],
    ],
    (False, 2, "quadratic-spectral", 5.0): [
        [0.04162240001980293, 0.04492394541844048],
        [0.044923945418440484, 0.06927676852467685],
    ],
    (False, 2, None, None): [
        [0.039138345414733856, 0.043442657226281864],
        [0.04344265722628187, 0.07565545821184429],
    ],
    (False, 3, "bartlett", 6.0): [
        [0.04423709317881465, 0.046824113134677804],
        [0.04682411313467782, 0.06854381149397759],
    ],
    (False, 3, "parzen", 8.0): [
        [0.04336648940210968, 0.04564683189023851],
        [0.045646831890238514, 0.06729957534327666],
    ],
    (False, 3, "quadratic-spectral", 5.0): [
        [0.04267623416857668, 0.04446896360658109],
        [0.04446896360658109, 0.06500997714809104],
    ],
    (False, 3, None, None): [
        [0.041834363425086576, 0.043248135102452005],
        [0.043248135102452005, 0.06475634312632854],
    ],
    (True, 1, "bartlett", 6.0): [
        [0.050591983275759526, 0.052756199758803475],
        [0.05275619975880347, 0.07837087151989267],
    ],
    (True, 1, "parzen", 8.0): [
        [0.04797943385296708, 0.05048692292160019],
        [0.05048692292160019, 0.07700954024542318],
    ],
    (True, 1, "quadratic-spectral", 5.0): [
        [0.046735578259553906, 0.048766189031658726],
        [0.048766189031658726, 0.07330409668857116],
    ],
    (True, 1, None, None): [
        [0.05991425431789259, 0.058811452903721896],
        [0.058811452903721896, 0.09158889572077024],
    ],
    (True, 2, "bartlett", 6.0): [
        [0.043244142629468685, 0.047292310643318555],
        [0.047292310643318555, 0.0729228997153076],
    ],
    (True, 2, "parzen", 8.0): [
        [0.04232102377838294, 0.04618722203132571],
        [0.04618722203132571, 0.07236481672318555],
    ],
    (True, 2, "quadratic-spectral", 5.0): [
        [0.04169194454782264, 0.04499900631304105],
        [0.04499900631304105, 0.0693925191821951],
    ],
    (True, 2, None, None): [
        [0.0392037394755772, 0.043515243119725946],
        [0.04351524311972595, 0.07578186666332605],
    ],
    (True, 3, "bartlett", 6.0): [
        [0.044311006450625544, 0.04690234891268062],
        [0.046902348912680615, 0.06865833749480296],
    ],
    (True, 3, "parzen", 8.0): [
        [0.04343894803101882, 0.045723100615201316],
        [0.04572310061520131, 0.06741202241987361],
    ],
    (True, 3, "quadratic-spectral", 5.0): [
        [0.04274753948882495, 0.04454326429765308],
        [0.04454326429765308, 0.06511859866379378],
    ],
    (True, 3, None, None): [
        [0.04190426211084278, 0.04332039597981617],
        [0.04332039597981617, 0.06486454085920462],
    ],
}
