"""
Table-driven tests of the input validation of the unit root tests

The arguments of ADF, DFGLS, PhillipsPerron, KPSS, VarianceRatio and ZivotAndrews are
checked when the test is created, and the statistic is still computed when it is first
used. The functions mackinnonp, mackinnoncrit, kpss_crit and auto_bandwidth check their
arguments when they are called.
"""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from arch.tests.validation_table import (
    BOOL_BAD,
    BOOL_GOOD,
    FLOAT_BAD,
    FLOAT_GOOD,
    INT_BAD,
    INT_GOOD,
    Row,
    bad_value_rows,
    good_value_rows,
    run_row,
    to_params,
)
from arch.unitroot import (
    ADF,
    DFGLS,
    KPSS,
    PhillipsPerron,
    VarianceRatio,
    ZivotAndrews,
    auto_bandwidth,
)
from arch.unitroot.unitroot import kpss_crit, mackinnoncrit, mackinnonp
from arch.utility.exceptions import InfeasibleTestException

NOBS = 100
Y = np.cumsum(np.random.default_rng(20231012).standard_normal(NOBS))


def adf(**kw):
    return ADF(**kw)


def dfgls(**kw):
    return DFGLS(**kw)


def pp(**kw):
    return PhillipsPerron(**kw)


def kpss(**kw):
    return KPSS(**kw)


def vr(**kw):
    return VarianceRatio(**kw)


def za(**kw):
    return ZivotAndrews(**kw)


def bandwidth(**kw):
    return auto_bandwidth(**kw)


def pvalue(**kw):
    return mackinnonp(**kw)


def critical_values(**kw):
    return mackinnoncrit(**kw)


def kpss_critical_values(**kw):
    return kpss_crit(**kw)


def without_none(bad):
    return [(value, exc) for value, exc in bad if value is not None]


NAN_Y = Y.copy()
NAN_Y[3] = np.nan
INF_Y = Y.copy()
INF_Y[7] = np.inf
Y_BAD = [
    (["a", "b", "c"] * 34, TypeError),
    (None, TypeError),
    (np.array([1 + 1j] * NOBS), TypeError),
    (NAN_Y, ValueError),
    (list(NAN_Y), ValueError),
    (pd.Series(NAN_Y), ValueError),
    (INF_Y, ValueError),
    (-INF_Y, ValueError),
    (np.ones((NOBS // 2, 2)), ValueError),
    (pd.DataFrame({"a": Y, "b": Y}), ValueError),
]
Y_GOOD = [
    Y,
    list(Y),
    Y.astype(np.float32),
    np.round(Y).astype(np.int64),
    pd.Series(Y),
    pd.DataFrame({"a": Y}),
    Y[:, None],
    Y[None, :],
]
LAGS_BAD = without_none(INT_BAD) + [(-1, ValueError), (np.int64(-3), ValueError)]
LAGS_GOOD = [0, 1, *INT_GOOD, None]
MAX_LAGS_BAD = LAGS_BAD
MAX_LAGS_GOOD = [0, 1, *INT_GOOD, None]
LOW_MEMORY_BAD = without_none(BOOL_BAD)
LOW_MEMORY_GOOD = [*BOOL_GOOD, None]
OPTION_BAD = [
    ("bogus", ValueError),
    ("", ValueError),
    (None, TypeError),
    (1, TypeError),
]
TRIM_BAD = [
    *FLOAT_BAD,
    (-0.01, ValueError),
    (0.34, ValueError),
    (0.5, ValueError),
    (np.float64(1.0), ValueError),
]
TRIM_GOOD = [
    0.15,
    0.3,
    1 / 3,
    np.float32(0.2),
    np.float64(0.1),
    Fraction(1, 4),
    1 / 100,
]
STAT_BAD = [(v, e) for v, e in FLOAT_BAD if not (isinstance(v, float) and np.isinf(v))]
STAT_GOOD = [-3.5, 0, 2, np.float32(-1.5), np.int64(-4), np.inf, -np.inf, 1.0e6]

RULE_INT = r"an integer .*"
RULE_FLOAT = r"a .*float.*"
RULE_BOOL = r"True or False.*"


def table_rows():
    rows = []

    # --- y and the options shared by the tests
    regression_tests = [
        ("ADF", adf, {"y": Y}),
        ("DFGLS", dfgls, {"y": Y}),
        ("PP", pp, {"y": Y}),
        ("KPSS", kpss, {"y": Y}),
        ("VR", vr, {"y": Y}),
        ("ZA", za, {"y": Y}),
    ]
    for target, func, base in regression_tests:
        rows += bad_value_rows(target, func, base, "y", Y_BAD)
        rows += good_value_rows(target, func, base, "y", Y_GOOD)
        rows += bad_value_rows(target, func, base, "trend", OPTION_BAD)
    rows += bad_value_rows("bandwidth", bandwidth, {"y": Y}, "y", Y_BAD)
    rows += good_value_rows("bandwidth", bandwidth, {"y": Y}, "y", Y_GOOD)
    rows += bad_value_rows("bandwidth", bandwidth, {"y": Y}, "kernel", OPTION_BAD)

    # --- lags
    for target, func in (
        ("ADF", adf),
        ("DFGLS", dfgls),
        ("PP", pp),
        ("ZA", za),
    ):
        base = {"y": Y}
        rows += bad_value_rows(target, func, base, "lags", LAGS_BAD, rule=RULE_INT)
        rows += good_value_rows(target, func, base, "lags", LAGS_GOOD)
    # KPSS accepts -1, which selects the lags with the old rule
    kpss_bad = without_none(INT_BAD) + [(-2, ValueError), (np.int64(-5), ValueError)]
    rows += bad_value_rows("KPSS", kpss, {"y": Y}, "lags", kpss_bad, rule=RULE_INT)
    rows += good_value_rows(
        "KPSS", kpss, {"y": Y}, "lags", [*LAGS_GOOD, -1, np.int32(-1)]
    )
    # VarianceRatio needs lags of at least 2 and has no None
    vr_bad = [*INT_BAD, (1, ValueError), (0, ValueError), (-2, ValueError)]
    rows += bad_value_rows("VR", vr, {"y": Y}, "lags", vr_bad, rule=RULE_INT)
    rows += good_value_rows("VR", vr, {"y": Y}, "lags", [2, 3, *INT_GOOD])

    # --- max_lags
    for target, func in (("ADF", adf), ("DFGLS", dfgls), ("ZA", za)):
        base = {"y": Y}
        rows += bad_value_rows(
            target, func, base, "max_lags", MAX_LAGS_BAD, rule=RULE_INT
        )
        rows += good_value_rows(target, func, base, "max_lags", MAX_LAGS_GOOD)
        rows += bad_value_rows(target, func, base, "method", OPTION_BAD)
    for target, func in (("ADF", adf), ("DFGLS", dfgls)):
        rows += bad_value_rows(
            target, func, {"y": Y}, "low_memory", LOW_MEMORY_BAD, rule=RULE_BOOL
        )
        rows += good_value_rows(target, func, {"y": Y}, "low_memory", LOW_MEMORY_GOOD)

    # --- ZivotAndrews.trim
    rows += bad_value_rows("ZA", za, {"y": Y}, "trim", TRIM_BAD, rule=RULE_FLOAT)
    rows += good_value_rows("ZA", za, {"y": Y}, "trim", TRIM_GOOD)

    # --- PhillipsPerron.test_type
    rows += bad_value_rows("PP", pp, {"y": Y}, "test_type", OPTION_BAD)

    # --- VarianceRatio flags
    for flag in ("debiased", "robust", "overlap"):
        rows += bad_value_rows("VR", vr, {"y": Y}, flag, BOOL_BAD, rule=RULE_BOOL)
        rows += good_value_rows("VR", vr, {"y": Y}, flag, BOOL_GOOD)

    # --- mackinnonp
    base = {"stat": -2.5}
    rows += bad_value_rows(
        "mackinnonp", pvalue, base, "stat", STAT_BAD, rule=RULE_FLOAT
    )
    rows += good_value_rows("mackinnonp", pvalue, base, "stat", STAT_GOOD)
    rows += bad_value_rows(
        "mackinnonp",
        pvalue,
        base,
        "num_unit_roots",
        without_none(INT_BAD),
        rule=RULE_INT,
    )
    rows += good_value_rows(
        "mackinnonp", pvalue, base, "num_unit_roots", [1, 6, *INT_GOOD[:3]]
    )
    rows += bad_value_rows("mackinnonp", pvalue, base, "regression", OPTION_BAD)
    rows += bad_value_rows("mackinnonp", pvalue, base, "dist_type", OPTION_BAD)
    for value in (0, -1, 7, 12, 13, np.int64(0), np.int8(-1)):
        rows.append(
            Row(
                "mackinnonp",
                pvalue,
                {**base, "num_unit_roots": value},
                ValueError,
                rf"^num_unit_roots must be an integer in \[1, 6\]; got {int(value)} ",
                id=f"mackinnonp-num_unit_roots-range-{type(value).__name__}{int(value)}",
                name="num_unit_roots",
            )
        )
    for dist in ("adf-z", "dfgls", "ADF-Z"):
        rows.append(
            Row(
                "mackinnonp",
                pvalue,
                {**base, "num_unit_roots": 2, "dist_type": dist},
                ValueError,
                r"^num_unit_roots must be 1 for dist_type=",
                id=f"mackinnonp-num_unit_roots-dist_type-{dist}",
                name="num_unit_roots",
            )
        )
    rows.append(
        Row(
            "mackinnonp",
            pvalue,
            {**base, "regression": "n", "dist_type": "dfgls"},
            ValueError,
            r"^regression must be one of 'c' or 'ct'; got 'n' ",
            id="mackinnonp-regression-dfgls",
            name="regression",
        )
    )

    # --- mackinnoncrit
    base = {"num_unit_roots": 1}
    nobs_bad = [
        (v, e) for v, e in FLOAT_BAD if not (isinstance(v, float) and v == np.inf)
    ] + [(0, ValueError), (-5, ValueError), (0.0, ValueError), (-np.inf, ValueError)]
    rows += bad_value_rows(
        "mackinnoncrit", critical_values, base, "nobs", nobs_bad, rule=RULE_FLOAT
    )
    rows += good_value_rows(
        "mackinnoncrit",
        critical_values,
        base,
        "nobs",
        [np.inf, np.float64(np.inf), 50, 50.5, np.int64(100), np.float32(60.5), 1e-3],
    )
    rows += bad_value_rows(
        "mackinnoncrit",
        critical_values,
        base,
        "num_unit_roots",
        without_none(INT_BAD),
        rule=RULE_INT,
    )
    rows += good_value_rows(
        "mackinnoncrit", critical_values, base, "num_unit_roots", [1, 12, *INT_GOOD[:3]]
    )
    rows += bad_value_rows(
        "mackinnoncrit", critical_values, base, "regression", OPTION_BAD
    )
    rows += bad_value_rows(
        "mackinnoncrit", critical_values, base, "dist_type", OPTION_BAD
    )
    for value in (0, -1, 13, 100, np.int64(0), np.int16(14)):
        rows.append(
            Row(
                "mackinnoncrit",
                critical_values,
                {"num_unit_roots": value},
                ValueError,
                rf"^num_unit_roots must be an integer in \[1, 12\]; got {int(value)} ",
                id=f"mackinnoncrit-num_unit_roots-range-{type(value).__name__}{int(value)}",
                name="num_unit_roots",
            )
        )
    rows.append(
        Row(
            "mackinnoncrit",
            critical_values,
            {"num_unit_roots": 2, "regression": "n"},
            ValueError,
            r"^num_unit_roots must be an integer in \[1, 1\]; got 2 ",
            id="mackinnoncrit-num_unit_roots-regression-n",
            name="num_unit_roots",
        )
    )
    for dist in ("adf-z", "dfgls"):
        rows.append(
            Row(
                "mackinnoncrit",
                critical_values,
                {"num_unit_roots": 3, "dist_type": dist},
                ValueError,
                r"^num_unit_roots must be 1 for dist_type=",
                id=f"mackinnoncrit-num_unit_roots-dist_type-{dist}",
                name="num_unit_roots",
            )
        )

    # --- kpss_crit
    base = {"stat": 0.3}
    kpss_stat_bad = [
        (v, e) for v, e in FLOAT_BAD if not (isinstance(v, float) and v == np.inf)
    ] + [(-0.01, ValueError), (-1, ValueError), (-np.inf, ValueError)]
    rows += bad_value_rows(
        "kpss_crit", kpss_critical_values, base, "stat", kpss_stat_bad, rule=RULE_FLOAT
    )
    rows += good_value_rows(
        "kpss_crit",
        kpss_critical_values,
        base,
        "stat",
        [0, 0.0, 0.3, 2, np.float32(0.5), np.int64(1), np.inf, *FLOAT_GOOD],
    )
    rows += bad_value_rows("kpss_crit", kpss_critical_values, base, "trend", OPTION_BAD)
    return rows


ROWS = table_rows()


@pytest.mark.parametrize("row", to_params(ROWS))
def test_validation(row):
    run_row(row)


# ------------------------------------------------------------------------------------
# Arguments that depend on the number of observations
# ------------------------------------------------------------------------------------
def nobs_rows():
    rows = []
    short = Y[:3]
    for target, func, kwargs in (
        ("ADF", adf, {"trend": "c"}),
        ("DFGLS", dfgls, {"trend": "c"}),
        ("PP", pp, {"trend": "c"}),
        ("KPSS", kpss, {"trend": "ct"}),
        ("ZA", za, {"trend": "c"}),
    ):
        rows.extend(
            Row(
                target,
                func,
                {"y": Y[:n], **kwargs},
                InfeasibleTestException,
                r"^y must be at least \d+ observations long to run the .* test "
                rf"with trend '{kwargs['trend']}'; got {n} observations \(ndarray\)$",
                id=f"{target}-short-{n}",
                name="y",
            )
            for n in (0, 1, 2)
        )
    rows.append(
        Row(
            "VR",
            vr,
            {"y": Y[:2]},
            InfeasibleTestException,
            r"^y must be at least 4 observations long to run the variance ratio test",
            id="VR-short-2",
            name="y",
        )
    )
    rows.append(
        Row(
            "ADF", adf, {"y": short}, InfeasibleTestException, r"at least 4", id="ADF-3"
        )
    )
    return rows


NOBS_ROWS = nobs_rows()


@pytest.mark.parametrize("row", to_params(NOBS_ROWS))
def test_validation_short_series(row):
    run_row(row)


def lags_boundary_cases():
    """(factory, nobs, lags) with the largest lags that the sample can support"""
    cases = []
    # ADF: the regression has nobs - 1 - lags rows and 1 + lags + k columns, and needs
    # one degree of freedom, so that nobs >= 3 + 2 lags + k
    for trend, k in (("n", 0), ("c", 1), ("ct", 2), ("ctt", 3)):
        cases.extend(
            pytest.param(
                lambda y, lags, trend=trend: ADF(y, lags=lags, trend=trend),
                3 + 2 * lags + k,
                lags,
                1,
                id=f"ADF-{trend}-{lags}",
            )
            for lags in (0, 1, 4, 10)
        )
    # DFGLS: the regression of the detrended data has no trend terms, and the trend
    # of the test needs 3 + k observations
    for trend, k in (("c", 1), ("ct", 2)):
        for lags in (0, 1, 4, 10):
            nobs = max(3 + k + lags, 3 + 2 * lags)
            cases.append(
                pytest.param(
                    lambda y, lags, trend=trend: DFGLS(y, lags=lags, trend=trend),
                    nobs,
                    lags,
                    nobs - 2 - 2 * lags,
                    id=f"DFGLS-{trend}-{lags}",
                )
            )
    return cases


@pytest.mark.parametrize(("factory", "nobs", "lags", "df"), lags_boundary_cases())
def test_lags_boundary_adf_dfgls(factory, nobs, lags, df):
    # The smallest sample is accepted and its regression has at least one degree of
    # freedom
    test = factory(Y[:nobs], lags)
    assert test.regression.df_resid == df
    assert df >= 1
    # One observation fewer cannot identify the regression
    with pytest.raises(InfeasibleTestException, match=r"^(lags|y) must be "):
        factory(Y[: nobs - 1], lags)


@pytest.mark.parametrize("nobs", [5, 10, 20])
def test_lags_boundary_pp_kpss(nobs):
    # The long-run covariance can use nobs - 1 residuals in PP and nobs in KPSS
    assert np.isfinite(PhillipsPerron(Y[:nobs], lags=nobs - 1).stat)
    assert np.isfinite(KPSS(Y[:nobs], lags=nobs).stat)
    with pytest.raises(
        InfeasibleTestException,
        match=rf"^lags must be at most {nobs - 1} for the {nobs} observations in y; "
        rf"got {nobs} \(int\)$",
    ):
        PhillipsPerron(Y[:nobs], lags=nobs)
    with pytest.raises(
        InfeasibleTestException,
        match=rf"^lags must be at most {nobs} for the {nobs} observations in y; "
        rf"got {nobs + 1} \(int\)$",
    ):
        KPSS(Y[:nobs], lags=nobs + 1)


@pytest.mark.filterwarnings("ignore::arch.utility.exceptions.InvalidLengthWarning")
@pytest.mark.parametrize(
    ("overlap", "debiased", "nobs"),
    [(True, True, 12), (True, False, 11), (False, True, 20), (False, False, 20)],
)
def test_lags_boundary_variance_ratio(overlap, debiased, nobs):
    lags = 10
    kwargs = {"lags": lags, "overlap": overlap, "debiased": debiased}
    stat = VarianceRatio(Y[:nobs], **kwargs).stat
    assert np.isfinite(stat)
    with pytest.raises(InfeasibleTestException, match=r"^lags must be at most "):
        VarianceRatio(Y[: nobs - 1], **kwargs)


@pytest.mark.parametrize(
    ("trend", "k", "basecols"), [("c", 1, 4), ("ct", 2, 5), ("t", 1, 4)]
)
def test_za_lags_and_trim(trend, k, basecols):
    # Nobs - 1 - lags rows and basecols + lags columns with one degree of freedom
    lags = 2
    extra = {"c": 1, "ct": 2, "t": 3}[trend]
    nobs = max(3 + k + lags, 2 + basecols + 2 * lags)
    # int(nobs * trim) must also be at least lags + 1 (c), + 2 (ct) or + 3 (t)
    trim_nobs = max(nobs, 3 * (lags + extra))
    assert ZivotAndrews(Y[:trim_nobs], lags=lags, trend=trend, trim=1 / 3) is not None
    with pytest.raises(InfeasibleTestException, match=r"^lags must be at most"):
        ZivotAndrews(Y[: nobs - 1], lags=lags, trend=trend, trim=1 / 3)
    nobs = NOBS
    needed = lags + extra
    trim_ok = needed / nobs + 1e-9
    trim_small = (needed - 1) / nobs + 1e-9
    assert np.isfinite(ZivotAndrews(Y, lags=lags, trend=trend, trim=trim_ok).stat)
    with pytest.raises(
        InfeasibleTestException,
        match=rf"^trim must be large enough that int\(nobs \* trim\) is at least "
        rf"{needed} for lags={lags} and trend '{trend}'",
    ):
        ZivotAndrews(Y, lags=lags, trend=trend, trim=trim_small)


@pytest.mark.parametrize("trim", [0, 0.0, 0.005])
def test_za_trim_too_small(trim):
    with pytest.raises(InfeasibleTestException, match=r"^trim must be large enough"):
        ZivotAndrews(Y, trim=trim)


@pytest.mark.parametrize(
    ("factory", "limit"),
    [
        (lambda m: ADF(Y, max_lags=m), (NOBS - 4) // 2),
        (lambda m: ADF(Y, max_lags=m, low_memory=True), (NOBS - 4) // 2),
        (lambda m: ADF(Y, max_lags=m, trend="n"), (NOBS - 3) // 2),
        (lambda m: DFGLS(Y, max_lags=m), (NOBS - 3) // 2),
        (lambda m: ZivotAndrews(Y, max_lags=m), (NOBS - 5) // 2),
    ],
    ids=["ADF", "ADF-low_memory", "ADF-n", "DFGLS", "ZA"],
)
def test_max_lags_against_nobs(factory, limit):
    assert factory(limit) is not None
    with pytest.raises(
        InfeasibleTestException,
        match=rf"^max_lags must be at most {limit} for the {NOBS} observations in y; "
        rf"got {limit + 1} \(int\)$",
    ):
        factory(limit + 1)
    with pytest.raises(InfeasibleTestException, match=r"^max_lags must be at most"):
        factory(10 * NOBS)


def test_max_lags_ignored_with_lags():
    # max_lags is not used if lags is given, so only its type is checked
    assert ADF(Y, lags=2, max_lags=10 * NOBS).lags == 2
    assert DFGLS(Y, lags=2, max_lags=10 * NOBS).lags == 2
    assert ZivotAndrews(Y, lags=2, max_lags=10 * NOBS).lags == 2
    with pytest.raises(TypeError, match=r"^max_lags must be an integer"):
        ADF(Y, lags=2, max_lags=2.5)


def test_default_lags_against_nobs():
    for func, nobs in ((PhillipsPerron, 5), (lambda y: KPSS(y, lags=-1), 4)):
        with pytest.raises(
            InfeasibleTestException,
            match=r"^y must be at least \d+ observations long .* and the default "
            rf"number of lags; got {nobs} observations \(ndarray\)$",
        ):
            func(Y[:nobs])


# ------------------------------------------------------------------------------------
# Other changes
# ------------------------------------------------------------------------------------
def test_checks_run_at_creation_and_statistic_is_lazy():
    test = ADF(Y)
    assert test._stat is None
    with pytest.raises(TypeError, match=r"^low_memory must be True or False or None"):
        ADF(Y, low_memory="yes")
    assert np.isfinite(test.stat)


def test_infeasible_test_exception_is_value_error_and_names_the_test():
    for func in (PhillipsPerron, KPSS, VarianceRatio):
        with pytest.raises(ValueError, match=r"^y must be at least") as info:
            func(Y[:1])
        assert isinstance(info.value, RuntimeError)
        assert "ADF" not in str(info.value)
    with pytest.raises(InfeasibleTestException, match="Phillips-Perron"):
        PhillipsPerron(Y[:1])
    with pytest.raises(InfeasibleTestException, match="KPSS"):
        KPSS(Y[:1])
    with pytest.raises(InfeasibleTestException, match="variance ratio"):
        VarianceRatio(Y[:1])


@pytest.mark.parametrize("dist_type", ["adf-t", "adf-z", "dfgls"])
def test_critical_values_are_copies(dist_type):
    # Changing the result must not change the table that later calls use
    first = critical_values(dist_type=dist_type)
    expected = first.copy()
    first[:] = 0.0
    assert np.all(critical_values(dist_type=dist_type) == expected)


def test_critical_value_arguments_are_not_indexed_from_the_end():
    # num_unit_roots=0 and -1 used to select the last row of the table
    for value in (0, -1):
        with pytest.raises(ValueError, match=r"^num_unit_roots must be"):
            mackinnonp(-3.0, num_unit_roots=value)
        with pytest.raises(ValueError, match=r"^num_unit_roots must be"):
            mackinnoncrit(value)
    assert 0.0 < mackinnonp(-3.0, num_unit_roots=6) < 1.0
    assert mackinnoncrit(12, "ct").shape == (3,)


def test_nan_and_inf_in_y_name_the_argument():
    for func in (ADF, DFGLS, PhillipsPerron, KPSS, VarianceRatio, ZivotAndrews):
        for bad in (NAN_Y, INF_Y):
            with pytest.raises(ValueError, match=r"^y must be a 1-d array of finite"):
                func(bad)
    with pytest.raises(ValueError, match=r"^y must be a 1-d array of finite"):
        auto_bandwidth(NAN_Y)


def test_valid_results_are_unchanged_by_numpy_scalars():
    expected = ADF(Y, lags=np.int64(2), max_lags=np.int32(5)).stat
    assert ADF(Y, lags=2, max_lags=5).stat == expected
    assert (
        ZivotAndrews(Y, trim=np.float32(0.25)).stat
        == ZivotAndrews(Y, trim=float(np.float32(0.25))).stat
    )
    assert mackinnonp(np.float32(-2.0)) == mackinnonp(float(np.float32(-2.0)))
    assert np.all(
        mackinnoncrit(np.int64(2), nobs=np.int64(100)) == mackinnoncrit(2, nobs=100)
    )
