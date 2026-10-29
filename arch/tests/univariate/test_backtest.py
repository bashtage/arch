"""
Tests of the VaR backtests: exceptions, Kupiec, Christoffersen and the traffic light.

The references are in the constants at the top.  They were produced by
``results/var_backtest_reference.R`` (R 4.6.1, rugarch 1.5.6), which prints them, and
are not computed at run time, so R is not needed.  Each case has

* the statistics and p-values of an independent implementation of the likelihoods in R,
  with the optimum found numerically and 0 log 0 = 0 (``uc``, ``ind``, ``cc``), and
* for the cases that it can compute, ``rugarch::VaRTest``, which gives the Kupiec
  and the conditional coverage tests (``rugarch_uc``, ``rugarch_cc``) and, by
  difference, the independence statistic (``rugarch_ind``).  rugarch fails if a state
  of the exceptions is never followed by another observation, so the edge cases only
  have the independent implementation.  The package GAS is not on CRAN any more.

The published tables are the Basel Committee on Banking Supervision (1996),
"Supervisory framework for the use of backtesting in conjunction with the internal
models approach to market risk capital requirements", Tables 1 and 2, and Kupiec (1995),
"Techniques for verifying the accuracy of risk measurement models", Table 1.
"""

import dataclasses
import math

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from scipy import stats

from arch.univariate import (
    TrafficLightResult,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
    traffic_light,
    var_exceptions,
)
from arch.utility.testing import WaldTestStatistic

# R 4.6.1, rugarch 1.5.6: results/var_backtest_reference.R.  Statistics and p-values
# are (statistic, p-value); ``pattern`` is the 0/1 exception series.


def _vec(text):
    return [float(v) for v in text.split()]


REFERENCE = {
    "returns_seed11": {
        "returns": _vec(
            "0.884 -0.414 -2.4664 0.2347 0.0737 -3.0993 0.2947 -1.3251 0.4447 "
            "0.1135 0.0811 0.24 0.8226 -0.0868 -1.2449 -0.8484 0.7954 1.0453 "
            "-0.7192 -0.5478 -0.2326 -1.0284 -0.7834 -0.6129 -0.6765 1.6207 2.7682 "
            "0.5525 0.4297 -0.205 -0.4566 -0.0323 3.6169 0.1993 -2.5005 0.5329 "
            "-0.9445 -2.1619 -0.5379 0.3664 -0.0233 0.7898 0.1659 0.3854 2.4349 "
            "-1.8065 0.4512 -0.6231 1.3388 -1.257 -0.4729 -1.3628 -1.7851 -0.2643 "
            "-0.0291 0.2896 0.5763 1.3619 -0.8864 -0.5708 0.8224 -0.223 -0.1742 "
            "-0.1889 0.6195 -0.5046 2.5886 1.7041 2.8514 -0.6379 -1.3921 0.8705 "
            "-0.7778 -0.1453 0.5057 -0.7628 -1.6801 -2.2573 0.1143 0.2312 -0.0166 "
            "1.6861 -0.5168 -0.5079 0.115 -0.1585 0.2148 -1.1213 -0.41 -0.0647 "
            "-0.6769 3.1862 2.6405 0.3335 1.4812 0.126 0.1164 0.2196 1.4142 0.2523 "
        ),
        "var": _vec(
            "1.6807 1.3165 1.9878 1.3343 1.4011 2.5724 1.4297 1.6975 2.4748 1.4781 "
            "1.5463 1.8959 2.5096 2.4357 2.2817 2.0707 1.9498 1.7509 1.5233 1.9477 "
            "1.584 2.2112 1.7945 1.7769 1.3975 1.9515 1.8411 1.3373 1.4805 1.8398 "
            "1.9802 1.747 1.8577 1.5822 2.3852 2.1607 1.6802 1.4521 1.6527 1.392 "
            "1.6416 1.5988 1.9654 2.1758 1.7488 2.4526 2.1551 1.334 2.0121 2.4138 "
            "1.9776 2.1025 1.876 1.7268 1.6078 1.8993 1.8587 2.3893 1.5299 2.1015 "
            "1.4932 2.3059 1.545 2.0789 2.3044 1.5564 1.3914 1.9719 1.5687 2.1314 "
            "2.2186 1.8758 1.8768 1.585 2.5513 2.0209 1.6718 1.7532 1.7949 1.6057 "
            "1.3828 1.9689 1.9845 1.6263 2.3867 1.8674 1.5677 2.1697 2.386 1.6126 "
            "1.7968 2.322 1.3348 1.9427 2.3826 1.6309 2.323 1.5698 2.0883 1.6095 "
        ),
        "p": 0.05,
        "nexc": 6,
        "uc": (0.19842212738650744, 0.6559974850549282),
        "ind": (0.9045292090106258, 0.341570159760567),
        "cc": (1.1029513363971333, 0.5760990517662391),
        "rugarch_uc": (0.19842212738650808, 0.6559974850549277),
        "rugarch_cc": (1.10295133639713, 0.5760990517662401),
        "rugarch_ind": 0.9045292090106218,
    },
    "returns_seed12": {
        "returns": _vec(
            "-0.0486 1.7221 1.1885 -1.8879 1.1485 1.4168 0.4466 -2.1406 0.6878 "
            "-0.4884 -0.6566 -0.2587 -1.5926 -2.9087 -1.7848 2.1219 -0.7422 -1.8782 "
            "1.0654 0.0395 -0.3577 -1.3986 -1.1487 4.0373 -0.7669 0.2991 -4.0451 "
            "-0.5916 1.8961 -0.532 1.9622 -3.2388 -3.6734 -1.5968 2.1451 -2.2469 "
            "0.663 -2.5204 1.2594 -0.8696 -0.8042 1.5445 0.1668 0.5016 0.4463 "
            "-0.5238 1.1337 -1.2747 2.8903 -0.5836 -0.6582 -1.0083 -0.6754 1.8632 "
            "2.5039 0.8497 1.1205 -1.6519 0.0474 2.8417 -1.8119 0.0369 -0.1448 "
            "-1.573 1.9883 1.0617 2.0927 -2.4037 1.7943 0.4095 -1.5912 -1.2715 "
            "-2.3672 1.4885 -1.1619 -0.3513 0.9994 1.1258 0.8649 -0.8706 1.3461 "
            "0.715 0.0936 0.9105 -0.8059 -1.3014 -1.5272 0.4324 1.009 0.7907 "
            "-0.8563 -0.1704 -1.301 1.2623 -4.7417 1.1483 -0.4258 -0.9853 -3.1823 "
            "-3.8061 "
        ),
        "var": _vec(
            "1.4072 2.3919 2.5563 1.6703 1.5388 1.3605 1.5511 2.1602 1.346 1.3269 "
            "1.8327 2.3868 1.811 1.8169 1.6644 1.894 1.9181 2.0274 2.1918 1.4642 "
            "1.6032 2.3526 1.4447 2.25 1.6026 1.6685 1.9801 1.5641 1.894 2.1974 "
            "1.6328 2.4913 2.4775 2.3872 2.1492 2.5543 2.2289 2.4262 1.822 1.8306 "
            "2.0903 2.014 2.6023 1.5759 2.4268 1.4409 1.8174 1.3762 1.5167 2.3354 "
            "1.8352 1.7916 1.87 1.7396 2.0424 2.5939 2.0501 2.0714 2.1599 2.4002 "
            "2.3026 1.8878 2.6066 2.0473 1.7031 1.5259 1.5031 2.3054 1.7827 2.3211 "
            "1.7287 2.2564 2.1163 1.8207 1.7314 2.5704 2.3523 2.2393 1.5233 2.3891 "
            "2.0288 2.4709 1.4789 1.6353 2.2612 1.6608 1.3886 2.4711 1.8144 1.4498 "
            "2.2021 1.519 1.532 1.7982 2.073 1.9674 2.3262 1.8051 1.7253 1.7682 "
        ),
        "p": 0.05,
        "nexc": 13,
        "uc": (9.536731078116802, 0.0020140012680092976),
        "ind": (1.4309927856447757, 0.2316024836761147),
        "cc": (10.967723863761577, 0.00415325908240721),
        "rugarch_uc": (9.536731078116814, 0.0020140012680093244),
        "rugarch_cc": (10.967723863761586, 0.004153259082407157),
        "rugarch_ind": 1.4309927856447722,
    },
    "returns_seed13": {
        "returns": _vec(
            "-1.2096 -0.1572 1.0893 -0.4188 0.283 -0.0455 0.35 -0.4315 -0.2127 "
            "0.5708 0.7252 0.7166 0.3387 1.1684 0.4522 0.6366 2.2888 0.2044 1.1099 "
            "0.1994 0.3843 -0.4027 -0.0542 0.5128 -0.658 0.4636 0.7322 -1.0248 "
            "-0.1284 -0.3202 1.4175 -1.4791 -0.6033 3.7912 -2.2923 1.8981 -1.2173 "
            "-1.2388 -2.7745 1.5287 1.2601 0.0482 0.3392 1.3389 3.0822 2.626 1.8266 "
            "-0.5872 -0.6297 -2.9909 0.3102 -1.1997 -0.1425 -2.4033 0.0106 -0.372 "
            "0.3105 -0.0442 0.7222 0.2423 0.1798 -1.7862 -0.9509 1.1283 1.6208 "
            "1.0395 1.4425 -0.1977 -0.3202 -0.419 1.0501 -0.0014 0.7895 -0.1922 "
            "0.1929 0.2342 -2.6813 -1.459 -0.0184 -0.5796 1.2802 0.9712 1.3666 "
            "1.1312 1.1101 0.0838 -0.4895 0.9403 -0.0378 0.7517 0.5931 -1.2251 "
            "0.2875 0.4306 -0.741 -1.2145 0.0267 0.2858 -0.3973 -1.4719 0.2681 "
            "-2.0032 1.2422 2.1468 2.0236 -1.5018 0.2224 -0.6301 -1.1922 1.0592 "
            "0.6117 0.1959 -1.191 1.2541 -2.9895 0.6596 0.1464 -0.894 0.0677 1.8266 "
        ),
        "var": _vec(
            "2.2507 1.6398 1.8286 1.4361 2.5819 1.3302 2.0715 2.3217 2.4651 1.37 "
            "2.1858 2.4717 2.4877 2.061 2.0969 1.7955 1.7861 2.0942 2.4548 2.2113 "
            "1.4963 2.0357 2.2079 2.0108 1.43 2.1317 1.3577 1.923 1.7501 2.1495 "
            "1.8727 1.8491 2.5249 2.3801 2.0266 1.457 1.914 1.3843 2.3143 1.4261 "
            "2.11 1.6185 2.5882 2.1113 2.1576 2.5252 1.5105 1.749 2.2796 2.2115 "
            "2.052 1.8394 1.411 2.629 1.344 1.9133 1.507 1.7646 1.6228 1.4437 "
            "1.9695 2.07 2.3709 2.1406 1.7773 1.9125 1.7097 1.4284 1.8322 1.8815 "
            "1.6743 1.7037 1.8023 1.5272 1.8483 2.5469 1.5713 1.9263 1.5771 2.1707 "
            "2.0265 1.9085 2.5582 1.8737 2.2561 2.0942 2.5617 1.8588 1.7365 2.6309 "
            "1.6236 2.5494 1.5174 2.537 1.3506 1.7399 2.1174 2.3808 2.5272 2.4288 "
            "2.1154 2.1102 2.3357 2.1349 1.7942 1.319 1.7008 1.4833 2.0957 2.237 "
            "1.8197 1.4381 1.5796 1.9345 2.363 1.5618 2.541 1.9131 2.3132 2.5323 "
        ),
        "p": 0.05,
        "nexc": 6,
        "uc": (0.0, 1.0),
        "ind": (0.6374678774139824, 0.4246292055031703),
        "cc": (0.6374678774139824, 0.7270689684843805),
        "rugarch_uc": (0.0, 1.0),
        "rugarch_cc": (0.6374678774139942, 0.7270689684843762),
        "rugarch_ind": 0.6374678774139942,
    },
    "iid_p01_n500": {
        "pattern": (
            "0000000000000000010000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000001000000000000000010000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000100000000000000"
            "0001000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000010000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000"
        ),
        "p": 0.01,
        "nexc": 6,
        "uc": (0.18988024532887948, 0.6630163069669712),
        "ind": (0.14604823027318048, 0.7023407829742428),
        "cc": (0.33592847560205996, 0.8453840669372774),
        "rugarch_uc": (0.18988024532887113, 0.6630163069669781),
        "rugarch_cc": (0.3359284756021027, 0.8453840669372594),
        "rugarch_ind": 0.14604823027323158,
    },
    "iid_p05_n250": {
        "pattern": (
            "0000000000000000100000000000100000000000100000100000000001000000"
            "0000000000000000000000100010000000000010000000000000001000100000"
            "0000010000000000000000000000000000000001001000000000001000000000"
            "1000000000010000001000100000000000000000000000000000110000"
        ),
        "p": 0.05,
        "nexc": 20,
        "uc": (4.039520476139188, 0.04444644930605593),
        "ind": (0.30613061740791636, 0.5800643566560412),
        "cc": (4.345651093547104, 0.11385545806554047),
        "rugarch_uc": (4.03952047613922, 0.044446449306055125),
        "rugarch_cc": (4.345651093547146, 0.1138554580655381),
        "rugarch_ind": 0.30613061740792613,
    },
    "clustered_p05_n400": {
        "pattern": (
            "0000000000000000000000000000000000000000000000000000001110000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000001000000000000000000000000000000011100000"
            "0000000000000000000000000000000011111000011100011111001000000001"
            "1000000000000000000000000000000001100000000000000000000001000000"
            "0000010000100000"
        ),
        "p": 0.05,
        "nexc": 28,
        "uc": (3.0120608060079803, 0.08264715476942851),
        "ind": (58.56427786513569, 1.9675661369854264e-14),
        "cc": (61.576338671143674, 4.2546902656531225e-14),
        "rugarch_uc": (3.012060806007948, 0.08264715476943041),
        "rugarch_cc": (61.57633867114364, 4.2521541843143495e-14),
        "rugarch_ind": 58.56427786513569,
    },
    "no_11_transitions": {
        "pattern": (
            "0000001000000100000010000001000000100000010000001000000100000010"
            "0000010000001000000100000010000001000000100000010000001000000100"
            "000010000001"
        ),
        "p": 0.1,
        "nexc": 20,
        "uc": (2.557358356885601, 0.1097818849920338),
        "ind": (6.3883809797155635, 0.011486975199763967),
        "cc": (8.945739336601164, 0.011414512940329493),
        "rugarch_uc": (2.5573583568856217, 0.10978188499203256),
        "rugarch_cc": (8.945739336601179, 0.011414512940329358),
        "rugarch_ind": 6.388380979715556,
    },
    "alternating": {
        "pattern": ("101010101010101010101010101010101010101010101010101010101010"),
        "p": 0.5,
        "nexc": 30,
        "uc": (0.0, 1.0),
        "ind": (81.77441734193008, 1.525403301423487e-19),
        "cc": (81.77441734193008, 1.749487701236992e-18),
        "rugarch_uc": (0.0, 1.0),
        "rugarch_cc": (81.77441734193008, 0.0),
        "rugarch_ind": 81.77441734193008,
    },
    "single_cluster": {
        "pattern": (
            "0000000000000000000000000000001111110000000000000000000000000000"
            "000000000000000000000000000000000000"
        ),
        "p": 0.05,
        "nexc": 6,
        "uc": (0.19842212738650744, 0.6559974850549282),
        "ind": (28.807968913528697, 7.992159595290531e-08),
        "cc": (29.006391040915204, 5.027385815892418e-07),
        "rugarch_uc": (0.19842212738650808, 0.6559974850549277),
        "rugarch_cc": (29.006391040915197, 5.027385815425944e-07),
        "rugarch_ind": 28.80796891352869,
    },
    "two_clusters_short": {
        "pattern": ("01111000000000000000110000000000000000000000000000"),
        "p": 0.1,
        "nexc": 6,
        "uc": (0.21024736654629805, 0.6465735897253202),
        "ind": (12.618183857267276, 0.00038201230771920015),
        "cc": (12.828431223813574, 0.0016381043138550494),
        "rugarch_uc": (0.21024736654629653, 0.6465735897253215),
        "rugarch_cc": (12.82843122381358, 0.0016381043138550266),
        "rugarch_ind": 12.618183857267283,
    },
    "single_exception": {
        "pattern": ("0000000000001000000000000"),
        "p": 0.04,
        "nexc": 1,
        "uc": (0.0, 1.0),
        "ind": (0.08698393898551515, 0.7680470508728161),
        "cc": (0.08698393898551515, 0.9574402427908988),
        "rugarch_uc": (0.0, 1.0),
        "rugarch_cc": (0.08698393898551571, 0.9574402427908985),
        "rugarch_ind": 0.08698393898551571,
    },
    "zero_exceptions": {
        "pattern": (
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000000000"
            "0000000000000000000000000000000000000000000000000000000000"
        ),
        "p": 0.01,
        "nexc": 0,
        "uc": (5.025167926750721, 0.024981503053449743),
        "ind": (0.0, 1.0),
        "cc": (5.025167926750721, 0.08105851616218143),
    },
    "all_exceptions": {
        "pattern": ("1111111111111111111111111111111111111111"),
        "p": 0.05,
        "nexc": 40,
        "uc": (239.65858188431926, 4.6682072578687354e-54),
        "ind": (0.0, 1.0),
        "cc": (239.65858188431926, 9.094947017729365e-53),
    },
    "first_only": {
        "pattern": ("10000000000000000000000000000000000000000000000000"),
        "p": 0.05,
        "nexc": 1,
        "uc": (1.2142960691147362, 0.2704829623685363),
        "ind": (0.0, 1.0),
        "cc": (1.2142960691147362, 0.5449026988068651),
    },
    "last_only": {
        "pattern": ("00000000000000000000000000000000000000000000000001"),
        "p": 0.05,
        "nexc": 1,
        "uc": (1.2142960691147362, 0.2704829623685363),
        "ind": (0.0, 1.0),
        "cc": (1.2142960691147362, 0.5449026988068651),
    },
    "two_obs": {
        "pattern": ("01"),
        "p": 0.5,
        "nexc": 1,
        "uc": (0.0, 1.0),
        "ind": (0.0, 1.0),
        "cc": (0.0, 1.0),
    },
    "two_obs_11": {
        "pattern": ("11"),
        "p": 0.3,
        "nexc": 2,
        "uc": (4.8158912173037445, 0.028198486433556325),
        "ind": (0.0, 1.0),
        "cc": (4.8158912173037445, 0.08999999999999998),
    },
}

# R 4.6.1 pbinom(0:15, 250, 0.01)
BASEL_TABLE2_CDF = [
    0.08105851616218145,
    0.28575173879395277,
    0.5431689733157261,
    0.7581166977648834,
    0.8921876269036251,
    0.9588168159301516,
    0.9862985521447962,
    0.9959746612881921,
    0.9989434675026432,
    0.9997498099312595,
    0.999946101370953,
    0.999989361192373,
    0.9999980641362445,
    0.9999996735384523,
    0.9999999487392628,
    0.9999999924748797,
]

# R 4.6.1 pbinom(0:15 - 1, 250, 0.01, lower.tail = FALSE): the type 1 error of Table 1
BASEL_TYPE1 = [
    1.0,
    0.9189414838378185,
    0.7142482612060472,
    0.45683102668427383,
    0.24188330223511653,
    0.10781237309637494,
    0.04118318406984842,
    0.013701447855203702,
    0.0040253387118078235,
    0.0010565324973568267,
    0.0002501900687405086,
    5.389862904704019e-05,
    1.0638807626992087e-05,
    1.9358637554504537e-06,
    3.264615476595364e-07,
    5.12607372364154e-08,
]

# (n, p, first yellow count, first red count): R 4.6.1 qbinom(0.95, n, p) and
# qbinom(0.9999, n, p), the definition of the zones for other sample sizes
BINOM_ZONES = [
    (250, 0.01, 5, 10),
    (250, 0.05, 18, 27),
    (500, 0.01, 9, 15),
    (1000, 0.01, 15, 24),
    (100, 0.05, 9, 15),
    (60, 0.05, 6, 11),
    (250, 0.001, 1, 4),
    (20, 0.001, 0, 2),
    (20, 0.5, 14, 18),
]

# R 4.6.1: the exceptions N with LR_uc <= qchisq(0.95, 1), as (min N, max N)
KUPIEC_REGIONS = {
    (0.01, 255): (1, 6),
    (0.01, 510): (2, 10),
    (0.01, 1000): (5, 16),
}

# Basel Committee (1996), Table 2: the cumulative probability in percent of the
# number of exceptions (0 to 9, and 10 or more) for 250 observations and 99% coverage
BASEL_TABLE2_PERCENT = [
    8.11,
    28.58,
    54.32,
    75.81,
    89.22,
    95.88,
    98.63,
    99.60,
    99.89,
    99.97,
    99.99,
]
# Basel Committee (1996), Table 1, 99% coverage, in percent, for 0 to 9 exceptions:
# the probability of exactly that number ("exact") and of that number or more ("type 1")
BASEL_TABLE1_EXACT = [8.1, 20.5, 25.7, 21.5, 13.4, 6.7, 2.7, 1.0, 0.3, 0.1]
BASEL_TABLE1_TYPE1 = [100.0, 91.9, 71.4, 45.7, 24.2, 10.8, 4.1, 1.4, 0.4, 0.1]
# Kupiec (1995), Table 1: the non-rejection region of the number of exceptions N of
# a 99% VaR at the 5% size, "N < 7", "1 < N < 11" and "4 < N < 17" for 255, 510 and
# 1000 observations.  The table has no lower limit for 255 observations.
KUPIEC_QUOTED = {
    255: (None, 7),
    510: (1, 11),
    1000: (4, 17),
}

CASES = list(REFERENCE)
RUGARCH_CASES = [name for name in CASES if "rugarch_cc" in REFERENCE[name]]
TOL = {"rtol": 1e-9, "atol": 1e-12}


def make_series(name):
    case = REFERENCE[name]
    if "pattern" in case:
        exceptions = np.array([int(c) for c in case["pattern"]], dtype=bool)
        returns = np.where(exceptions, -1.5, 0.5)
        var = np.ones(exceptions.shape[0])
    else:
        returns = np.array(case["returns"])
        var = np.array(case["var"])
    return returns, var, case["p"]


def series_with(exceptions):
    """Returns and VaR (1.0) whose exceptions are the 0/1 or bool sequence."""
    flags = np.asarray(exceptions, dtype=bool)
    return np.where(flags, -1.5, 0.5), np.ones(flags.shape[0])


def assert_stat(test, expected, **kwargs):
    stat, pval = expected
    tol = {**TOL, **kwargs}
    assert_allclose(test.stat, stat, **tol)
    assert_allclose(test.pval, pval, **tol)


def test_reference_cases_cover_the_edges():
    # Guard the table: the cases that matter are in it and rugarch is used for most
    assert len(CASES) >= 17
    assert len(RUGARCH_CASES) >= 10
    assert {"zero_exceptions", "all_exceptions", "no_11_transitions"} <= set(CASES)
    assert any("pattern" not in REFERENCE[name] for name in CASES)


@pytest.mark.parametrize("name", CASES)
def test_var_exceptions_count(name):
    returns, var, _ = make_series(name)
    exceptions = var_exceptions(returns, var)
    assert exceptions.dtype == np.bool_
    assert exceptions.shape == returns.shape
    assert int(exceptions.sum()) == REFERENCE[name]["nexc"]
    assert_array_equal(exceptions, -returns > var)


@pytest.mark.parametrize("name", CASES)
def test_kupiec_pof_reference(name):
    returns, var, p = make_series(name)
    assert_stat(kupiec_pof(returns, var, p), REFERENCE[name]["uc"])


@pytest.mark.parametrize("name", RUGARCH_CASES)
def test_kupiec_pof_rugarch(name):
    returns, var, p = make_series(name)
    assert_stat(kupiec_pof(returns, var, p), REFERENCE[name]["rugarch_uc"])


@pytest.mark.parametrize("name", CASES)
def test_christoffersen_independence_reference(name):
    returns, var, _ = make_series(name)
    assert_stat(christoffersen_independence(returns, var), REFERENCE[name]["ind"])


@pytest.mark.parametrize("name", RUGARCH_CASES)
def test_christoffersen_independence_rugarch(name):
    # rugarch has no independence test: its cc statistic less its uc statistic
    returns, var, _ = make_series(name)
    test = christoffersen_independence(returns, var)
    assert_allclose(test.stat, REFERENCE[name]["rugarch_ind"], **TOL)


@pytest.mark.parametrize("name", CASES)
def test_conditional_coverage_reference(name):
    returns, var, p = make_series(name)
    assert_stat(
        christoffersen_conditional_coverage(returns, var, p), REFERENCE[name]["cc"]
    )


@pytest.mark.parametrize("name", RUGARCH_CASES)
def test_conditional_coverage_rugarch(name):
    returns, var, p = make_series(name)
    test = christoffersen_conditional_coverage(returns, var, p)
    assert_stat(test, REFERENCE[name]["rugarch_cc"])


@pytest.mark.parametrize("name", CASES)
def test_conditional_coverage_is_the_sum(name):
    returns, var, p = make_series(name)
    cc = christoffersen_conditional_coverage(returns, var, p)
    parts = (
        kupiec_pof(returns, var, p).stat
        + christoffersen_independence(returns, var).stat
    )
    assert_allclose(cc.stat, parts, rtol=1e-12, atol=1e-12)


def test_wald_test_statistic_fields():
    returns, var, p = make_series("iid_p05_n250")
    tests = {
        "uc": (kupiec_pof(returns, var, p), 1),
        "ind": (christoffersen_independence(returns, var), 1),
        "cc": (christoffersen_conditional_coverage(returns, var, p), 2),
    }
    names = set()
    for test, df in tests.values():
        assert isinstance(test, WaldTestStatistic)
        assert test.df == df
        assert test.dist_name == f"chi2({df})"
        assert isinstance(test.stat, float)
        assert test.stat >= 0.0
        assert_allclose(test.pval, stats.chi2.sf(test.stat, df), rtol=1e-9, atol=1e-12)
        assert_allclose(
            list(test.critical_values.values()), stats.chi2.ppf([0.9, 0.95, 0.99], df)
        )
        assert list(test.critical_values) == ["10%", "5%", "1%"]
        assert test.null
        assert test.alternative
        assert test.null != test.alternative
        names.add(str(test).split(chr(10))[0])
    assert len(names) == 3
    assert "0.05" in tests["uc"][0].null
    assert "0.05" in tests["cc"][0].null
    assert "independent" in tests["ind"][0].null
    assert "Kupiec" in str(tests["uc"][0])
    assert "Christoffersen" in str(tests["ind"][0])
    assert "Christoffersen" in str(tests["cc"][0])


# Edge cases: closed forms of the limits


@pytest.mark.parametrize("p", [0.001, 0.01, 0.05, 0.3, 0.5])
@pytest.mark.parametrize("nobs", [2, 25, 250])
def test_no_exceptions(p, nobs):
    # 0 log 0 = 0, so LR_uc = -2 T log(1 - p); no exception means n01 = n11 = 0
    returns, var = series_with(np.zeros(nobs))
    uc = kupiec_pof(returns, var, p)
    assert_allclose(uc.stat, -2.0 * nobs * math.log1p(-p), rtol=1e-12)
    ind = christoffersen_independence(returns, var)
    assert ind.stat == 0.0
    assert ind.pval == 1.0
    cc = christoffersen_conditional_coverage(returns, var, p)
    assert_allclose(cc.stat, uc.stat, rtol=1e-12)


@pytest.mark.parametrize("p", [0.001, 0.01, 0.05, 0.3, 0.5])
@pytest.mark.parametrize("nobs", [2, 25, 250])
def test_all_exceptions(p, nobs):
    # LR_uc = -2 T log(p); every transition is 1 -> 1, so n00 = n01 = n10 = 0
    returns, var = series_with(np.ones(nobs))
    uc = kupiec_pof(returns, var, p)
    assert_allclose(uc.stat, -2.0 * nobs * math.log(p), rtol=1e-12)
    ind = christoffersen_independence(returns, var)
    assert ind.stat == 0.0
    assert ind.pval == 1.0
    cc = christoffersen_conditional_coverage(returns, var, p)
    assert_allclose(cc.stat, uc.stat, rtol=1e-12)


def test_exceptions_equal_to_coverage_have_zero_statistic():
    # 5 exceptions in 100 observations and p = 0.05
    flags = np.zeros(100, dtype=bool)
    flags[[3, 20, 41, 60, 88]] = True
    returns, var = series_with(flags)
    uc = kupiec_pof(returns, var, 0.05)
    assert_allclose(uc.stat, 0.0, atol=1e-12)
    assert uc.stat >= 0.0
    assert_allclose(uc.pval, 1.0, atol=1e-12)


def test_hand_computed_christoffersen():
    # 0 0 1 1 0 0 0 1 0 0 1 1 1 0 0: n00 = 5, n01 = 3, n10 = 3, n11 = 3 (14 transitions)
    flags = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0])
    n00, n01, n10, n11 = 5, 3, 3, 3
    assert sum([n00, n01, n10, n11]) == flags.shape[0] - 1
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    unrestricted = (1 - pi01) ** n00 * pi01**n01 * (1 - pi11) ** n10 * pi11**n11
    restricted = (1 - pi) ** (n00 + n10) * pi ** (n01 + n11)
    expected_ind = -2.0 * math.log(restricted / unrestricted)
    returns, var = series_with(flags)
    ind = christoffersen_independence(returns, var)
    assert_allclose(ind.stat, expected_ind, rtol=1e-12)
    assert_allclose(ind.pval, math.erfc(math.sqrt(expected_ind / 2.0)), rtol=1e-10)
    # Kupiec: 6 exceptions in 15 observations and p = 0.2
    x, nobs, p = 6, 15, 0.2
    expected_uc = -2.0 * math.log(
        ((1 - p) ** (nobs - x) * p**x)
        / ((1 - x / nobs) ** (nobs - x) * (x / nobs) ** x)
    )
    assert_allclose(kupiec_pof(returns, var, p).stat, expected_uc, rtol=1e-12)
    cc = christoffersen_conditional_coverage(returns, var, p)
    assert_allclose(cc.stat, expected_uc + expected_ind, rtol=1e-12)


def test_independence_two_observations():
    # Only the transition is known, and a single transition is explained exactly
    for flags in ([0, 0], [0, 1], [1, 0], [1, 1]):
        returns, var = series_with(flags)
        assert christoffersen_independence(returns, var).stat == 0.0


def test_single_observation():
    returns, var = series_with([1])
    assert_allclose(kupiec_pof(returns, var, 0.5).stat, -2.0 * math.log(0.5))
    returns, var = series_with([0])
    assert_allclose(kupiec_pof(returns, var, 0.5).stat, -2.0 * math.log(0.5))
    with pytest.raises(ValueError, match="at least 2 observations"):
        christoffersen_independence(returns, var)
    with pytest.raises(ValueError, match="at least 2 observations"):
        christoffersen_conditional_coverage(returns, var, 0.5)


def test_independence_ignores_order_of_clusters_only():
    # The Kupiec statistic only depends on the count, the independence statistic on
    # the order: clustered exceptions reject, spread out exceptions do not
    clustered = np.zeros(200, dtype=bool)
    clustered[50:58] = True
    spread = np.zeros(200, dtype=bool)
    spread[::25] = True
    c_ret, c_var = series_with(clustered)
    s_ret, s_var = series_with(spread)
    assert clustered.sum() == spread.sum()
    assert_allclose(
        kupiec_pof(c_ret, c_var, 0.04).stat, kupiec_pof(s_ret, s_var, 0.04).stat
    )
    assert christoffersen_independence(c_ret, c_var).stat > 10.0
    assert christoffersen_independence(s_ret, s_var).stat < 3.0


def test_exception_is_strict():
    # A loss equal to the VaR is not an exception
    returns = np.array([-1.0, -1.0000001, -0.9999999, 1.0, 0.0])
    var = np.ones(5)
    assert_array_equal(var_exceptions(returns, var), [False, True, False, False, False])


def test_sign_convention():
    returns = np.array([-3.0, 3.0, -1.0, 0.5])
    # VaR is a positive loss: a 2.5 VaR is exceeded by the return -3
    assert_array_equal(
        var_exceptions(returns, np.full(4, 2.5)), [True, False, False, False]
    )
    # A negative VaR (a quantile of the returns) is a gain that is always exceeded
    assert var_exceptions(returns, np.full(4, -2.5)).sum() == 3


def test_time_varying_var():
    returns = np.array([-1.0, -1.0, -3.0, 0.1])
    var = np.array([0.5, 2.0, 2.5, 0.1])
    assert_array_equal(var_exceptions(returns, var), [True, False, True, False])


def test_input_types():
    returns, var, p = make_series("returns_seed11")
    base = kupiec_pof(returns, var, p).stat
    for ret, v in (
        (list(returns), list(var)),
        (returns.astype(np.float32), var.astype(np.float32)),
        (pd.Series(returns), var),
        (returns, pd.Series(var)),
        (pd.DataFrame(returns), pd.DataFrame(var)),
        (returns.reshape(-1, 1), var.reshape(1, -1)),
    ):
        assert_allclose(kupiec_pof(ret, v, p).stat, base, rtol=1e-6)
    # Integers
    iret = np.array([-3, 1, 2, -1, 0, -5])
    ivar = np.array([2, 2, 2, 2, 2, 2])
    assert_array_equal(
        var_exceptions(iret, ivar), [True, False, False, False, False, True]
    )


# Alignment of pandas inputs


def _series_pair():
    rng = np.random.default_rng(20261018)
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    returns = pd.Series(rng.standard_normal(120), index=index, name="r")
    var = pd.Series(1.0 + 0.5 * rng.random(120), index=index, name="v")
    return returns, var


def test_pandas_same_index():
    returns, var = _series_pair()
    expected = var_exceptions(returns.to_numpy(), var.to_numpy())
    assert isinstance(expected, np.ndarray)
    result = var_exceptions(returns, var)
    assert isinstance(result, pd.Series)
    assert result.name == "exception"
    assert result.dtype == bool
    assert result.index.equals(returns.index)
    assert_array_equal(result.to_numpy(), expected)


def test_pandas_output_index():
    returns, var = _series_pair()
    expected = var_exceptions(returns.to_numpy(), var.to_numpy())
    # One pandas input is matched by position and gives its index
    for ret, v, index in (
        (returns, var.to_numpy(), returns.index),
        (returns.to_numpy(), var, var.index),
        (returns.to_frame(), var.to_frame(), returns.index),
        (returns.to_frame(), var.to_numpy(), returns.index),
        (returns.to_numpy(), var.to_frame(), var.index),
    ):
        result = var_exceptions(ret, v)
        assert isinstance(result, pd.Series)
        assert result.index.equals(index)
        assert_array_equal(result.to_numpy(), expected)
    # Lists and arrays give an array
    assert isinstance(var_exceptions(list(returns), list(var)), np.ndarray)


def test_pandas_alignment_on_index():
    returns, var = _series_pair()
    # var covers the observations 20 to 99 only, and in reverse order
    shifted = var.iloc[20:100].iloc[::-1]
    aligned_returns = returns.iloc[20:100].to_numpy()
    aligned_var = var.iloc[20:100].to_numpy()
    expected = var_exceptions(aligned_returns, aligned_var)
    result = var_exceptions(returns, shifted)
    assert_array_equal(result, expected)
    # The index is that of the common observations, in the order of returns
    assert result.index.equals(returns.index[20:100])
    reverse = var_exceptions(shifted, returns)
    assert_array_equal(reverse, expected[::-1])
    assert reverse.index.equals(shifted.index)
    assert_allclose(
        kupiec_pof(returns, shifted, 0.1).stat,
        kupiec_pof(aligned_returns, aligned_var, 0.1).stat,
    )
    assert_allclose(
        christoffersen_independence(returns, shifted).stat,
        christoffersen_independence(aligned_returns, aligned_var).stat,
    )
    assert_allclose(
        christoffersen_conditional_coverage(returns, shifted, 0.1).stat,
        christoffersen_conditional_coverage(aligned_returns, aligned_var, 0.1).stat,
    )


def test_pandas_alignment_keeps_the_order_of_returns():
    returns, var = _series_pair()
    shifted = var.iloc[::-1]
    # All the observations are in both, in the order of returns
    assert_array_equal(
        var_exceptions(returns, shifted),
        var_exceptions(returns.to_numpy(), var.to_numpy()),
    )


def test_pandas_alignment_beats_position():
    # Same length but different dates: the values are matched on the index, not by
    # position.  The overlap is the last 70 returns and the first 70 VaR values
    returns, var = _series_pair()
    returns = returns.iloc[:100]
    var = var.iloc[20:120]
    assert len(returns) == len(var)
    common = returns.index.intersection(var.index)
    expected = var_exceptions(
        returns.loc[common].to_numpy(), var.loc[common].to_numpy()
    )
    result = var_exceptions(returns, var)
    assert result.shape == (len(common),)
    assert_array_equal(result, expected)
    # and it differs from the match by position
    by_position = var_exceptions(returns.to_numpy(), var.to_numpy())
    assert by_position.shape != result.shape


def test_pandas_mixed_matches_by_position():
    returns, var = _series_pair()
    other = var.copy()
    other.index = pd.date_range("1999-01-01", periods=120, freq="D")
    expected = var_exceptions(returns.to_numpy(), var.to_numpy())
    # a Series and an array: by position, the index is not used
    assert_array_equal(var_exceptions(returns, other.to_numpy()), expected)
    assert_array_equal(var_exceptions(returns.to_numpy(), other), expected)


def test_pandas_disjoint_index():
    returns, var = _series_pair()
    other = var.copy()
    other.index = pd.date_range("1999-01-01", periods=120, freq="D")
    with pytest.raises(
        ValueError, match="returns must be aligned with var on at least 1"
    ):
        var_exceptions(returns, other)
    with pytest.raises(ValueError, match="at least 2 observations; got 1 "):
        christoffersen_independence(returns.iloc[:5], var.iloc[4:9])


def test_pandas_duplicated_index():
    returns, var = _series_pair()
    duplicated = var.copy()
    duplicated.index = duplicated.index[:-1].append(duplicated.index[:1])
    # Equal indices match by position, even with duplicates
    both = pd.Series(returns.to_numpy(), index=duplicated.index)
    assert_array_equal(
        var_exceptions(both, duplicated),
        var_exceptions(returns.to_numpy(), var.to_numpy()),
    )
    with pytest.raises(ValueError, match=r"index without duplicates.*\(Series\)"):
        var_exceptions(returns, duplicated)


def test_length_mismatch():
    returns, var = _series_pair()
    with pytest.raises(
        ValueError,
        match=r"var must be an array with the same length as returns \(120\)",
    ):
        var_exceptions(returns.to_numpy(), var.to_numpy()[:-1])
    with pytest.raises(ValueError, match="same length as returns"):
        kupiec_pof(returns, var.to_numpy()[:-1], 0.05)
    with pytest.raises(ValueError, match="same length as returns"):
        traffic_light(returns.to_numpy()[:-3], var, 0.05, window=20)


# Kupiec (1995)


@pytest.mark.parametrize("nobs", [255, 510, 1000])
def test_kupiec_nonrejection_regions(nobs):
    flags = np.zeros(nobs, dtype=bool)
    nonrejected = []
    var = np.ones(nobs)
    for count in range(nobs + 1):
        flags[:] = False
        flags[:count] = True
        returns = np.where(flags, -1.5, 0.5)
        if kupiec_pof(returns, var, 0.01).pval > 0.05:
            nonrejected.append(count)
    low, high = min(nonrejected), max(nonrejected)
    assert nonrejected == list(range(low, high + 1))
    # R: the same region
    assert (low, high) == KUPIEC_REGIONS[(0.01, nobs)]
    # Kupiec (1995), Table 1: lower < N < upper
    quoted_low, quoted_high = KUPIEC_QUOTED[nobs]
    assert high == quoted_high - 1
    if quoted_low is not None:
        assert low == quoted_low + 1


# Traffic light


def _window_series(count, window, history=40):
    """``count`` exceptions in the last ``window`` observations, after a history of
    exceptions that must not be counted."""
    flags = np.zeros(history + window, dtype=bool)
    flags[:history] = True
    flags[history : history + count] = True
    # A permutation of the window does not change the count
    flags[history:] = np.random.default_rng(count).permutation(flags[history:])
    return series_with(flags)


@pytest.mark.parametrize("count", range(11))
def test_traffic_light_basel_table2(count):
    returns, var = _window_series(count, 250)
    result = traffic_light(returns, var)
    assert isinstance(result, TrafficLightResult)
    assert result.exceptions == count
    assert result.nobs == 250
    assert result.p == 0.01
    zone = "green" if count <= 4 else ("yellow" if count <= 9 else "red")
    assert result.zone == zone
    # Plain Python numbers, not NumPy scalars (np.float64 is a float) or bool
    assert isinstance(result.exceptions, int)
    assert not isinstance(result.exceptions, (bool, np.integer))
    for value in (result.probability, result.type1_error):
        assert isinstance(value, float)
        assert not isinstance(value, np.floating)
    # the published percentages are rounded to 2 digits
    assert abs(100.0 * result.probability - BASEL_TABLE2_PERCENT[count]) <= 0.005 + 1e-9
    assert_allclose(result.probability, BASEL_TABLE2_CDF[count], rtol=1e-12)
    assert_allclose(result.probability, stats.binom.cdf(count, 250, 0.01), rtol=1e-12)


def test_traffic_light_red_beyond_table():
    for count in (10, 11, 15, 40, 250):
        returns, var = _window_series(count, 250, history=0)
        result = traffic_light(returns, var)
        assert result.zone == "red"
        assert result.exceptions == count
        assert_allclose(
            result.probability, stats.binom.cdf(count, 250, 0.01), rtol=1e-12
        )


def test_basel_table1():
    # The probability of exactly x and of x or more exceptions are the published
    # values of Table 1, from the probabilities of this function
    previous = 0.0
    for count in range(10):
        returns, var = _window_series(count, 250, history=0)
        result = traffic_light(returns, var)
        exact = result.probability - previous
        assert abs(100.0 * exact - BASEL_TABLE1_EXACT[count]) <= 0.05 + 1e-9
        assert (
            abs(100.0 * result.type1_error - BASEL_TABLE1_TYPE1[count]) <= 0.05 + 1e-9
        )
        previous = result.probability


@pytest.mark.parametrize("count", range(16))
def test_traffic_light_type1_error(count):
    returns, var = _window_series(count, 250)
    result = traffic_light(returns, var)
    assert_allclose(result.type1_error, BASEL_TYPE1[count], rtol=1e-12)
    # The two probabilities overlap by the probability of exactly count exceptions
    exact = BASEL_TABLE2_CDF[count] - (BASEL_TABLE2_CDF[count - 1] if count else 0.0)
    assert_allclose(result.probability + result.type1_error, 1.0 + exact, rtol=1e-12)


def test_traffic_light_type1_error_other_sizes():
    for count, window, p in (
        (0, 60, 0.05),
        (3, 60, 0.05),
        (12, 500, 0.01),
        (2, 20, 0.5),
    ):
        returns, var = _window_series(count, window)
        result = traffic_light(returns, var, p=p, window=window)
        expected = 1.0 - stats.binom.cdf(count - 1, window, p)
        assert_allclose(result.type1_error, expected, rtol=1e-10)


def _expected_zone(count, first_yellow, first_red):
    if count == 0 or count < first_yellow:
        return "green"
    return "yellow" if count < first_red else "red"


@pytest.mark.parametrize(("nobs", "p", "first_yellow", "first_red"), BINOM_ZONES)
def test_traffic_light_other_sample_sizes(nobs, p, first_yellow, first_red):
    # The zones begin where the cumulative probability reaches 95% and 99.99%
    last = min(nobs, first_red + 2)
    for count in range(last + 1):
        returns, var = _window_series(count, nobs, history=3)
        result = traffic_light(returns, var, p=p, window=nobs)
        assert result.exceptions == count
        assert result.zone == _expected_zone(count, first_yellow, first_red), count
        assert_allclose(result.probability, stats.binom.cdf(count, nobs, p), rtol=1e-12)
        assert result.nobs == nobs
        assert result.p == p


def test_traffic_light_no_exceptions_is_green():
    # window * p small: P(X <= 0) is above 95% but nothing is understated
    returns, var = series_with(np.zeros(20))
    result = traffic_light(returns, var, p=0.001, window=20)
    assert result.zone == "green"
    assert result.exceptions == 0
    assert result.probability > 0.95


def test_traffic_light_window():
    flags = np.zeros(400, dtype=bool)
    flags[:100] = True  # old exceptions: not counted
    flags[300:306] = True  # six exceptions in the last 250
    returns, var = series_with(flags)
    result = traffic_light(returns, var)
    assert result.exceptions == 6
    assert result.zone == "yellow"
    assert traffic_light(returns, var, window=100).exceptions == 6
    assert traffic_light(returns, var, window=100).nobs == 100
    assert traffic_light(returns, var, window=400).exceptions == 106
    assert traffic_light(returns, var, window=400).zone == "red"
    # Fewer observations than the window raises
    with pytest.raises(ValueError, match=r"window must be an integer no larger than"):
        traffic_light(returns, var, window=401)
    with pytest.raises(ValueError, match=r"observations \(399\); got 400 \(int\)"):
        traffic_light(returns[1:], var[1:], window=400)
    short = traffic_light(returns[:120], var[:120], window=120)
    assert short.nobs == 120
    assert short.exceptions == 100


def test_traffic_light_short_series_raises():
    returns, var = series_with(np.zeros(100))
    with pytest.raises(ValueError, match=r"window must be .*\(100\); got 250 \(int\)"):
        traffic_light(returns, var)


def test_traffic_light_result_is_immutable():
    returns, var = series_with(np.zeros(250))
    result = traffic_light(returns, var)
    assert dataclasses.is_dataclass(result)
    assert [f.name for f in dataclasses.fields(result)] == [
        "zone",
        "exceptions",
        "probability",
        "type1_error",
        "nobs",
        "p",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.zone = "red"
    assert result == traffic_light(returns, var)


def test_traffic_light_pandas_alignment():
    rng = np.random.default_rng(7)
    index = pd.date_range("2019-01-01", periods=300, freq="D")
    returns = pd.Series(rng.standard_normal(300), index=index)
    var = pd.Series(np.full(300, 2.0), index=index).iloc[40:]
    result = traffic_light(returns, var, window=100)
    expected = traffic_light(returns.to_numpy()[40:], var.to_numpy(), window=100)
    assert result == expected


def test_p_boundaries():
    returns, var = series_with([0, 1, 0, 0])
    for func in (kupiec_pof, christoffersen_conditional_coverage):
        # 0 is not a probability of an exception, and a confidence level such as
        # 0.95 or 0.99 is not one either
        for p in (0.0, 0.5 + 1e-12, 0.95, 0.99, 1.0):
            with pytest.raises(
                ValueError, match=r"p must be a finite float in \(0, 0.5\]"
            ):
                func(returns, var, p)
    # Values that are close to the lower boundary and 0.5 are valid
    assert kupiec_pof(returns, var, 1e-12).stat > 0.0
    assert kupiec_pof(returns, var, 0.5).stat > 0.0
