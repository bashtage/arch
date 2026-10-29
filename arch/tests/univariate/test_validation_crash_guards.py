"""
The arrays that volatility processes pass to the compiled recursions are checked first.

Before the checks, ``compute_variance`` with ``None`` for ``parameters``, ``sigma2`` or
``var_bounds`` ended the interpreter, an empty ``sigma2`` corrupted the heap, and arrays
of the wrong length were read or written out of bounds without an error.
"""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from arch.tests.validation_table import Row, run_row, to_params
from arch.univariate import (
    APARCH,
    ARCH,
    EGARCH,
    FIAPARCH,
    FIGARCH,
    GARCH,
    HARCH,
    NGARCH,
    ConstantVariance,
    EWMAVariance,
    FixedVariance,
    MIDASHyperbolic,
    RiskMetrics2006,
)

NOBS = 60

# The key is used in the ids of the tests
PROCESSES = {
    "garch": lambda: GARCH(1, 1, 1),
    "garch-power": lambda: GARCH(2, 0, 1, power=1.5),
    "arch": lambda: ARCH(2),
    "harch": lambda: HARCH([1, 5]),
    "midas": MIDASHyperbolic,
    "midas-asym": lambda: MIDASHyperbolic(asym=True),
    "ewma-estimated": lambda: EWMAVariance(None),
    "ewma-fixed": lambda: EWMAVariance(0.94),
    "rm2006": RiskMetrics2006,
    "egarch": lambda: EGARCH(1, 1, 1),
    "figarch": FIGARCH,
    "fiaparch": FIAPARCH,
    "aparch": lambda: APARCH(1, 1, 1),
    "ngarch": lambda: NGARCH(1, 1, 1),
    "ngarch-p2": lambda: NGARCH(2, 1, 2),
    "constant": ConstantVariance,
    "fixed": lambda: FixedVariance(np.linspace(0.5, 1.5, NOBS)),
}


def read_only(array):
    array = array.copy()
    array.flags.writeable = False
    return array


def setup_process(key):
    """A process and valid arguments for compute_variance."""
    process = PROCESSES[key]()
    process.start = 0
    process.stop = NOBS
    resids = np.random.default_rng(20241017).standard_normal(NOBS)
    args = {
        "parameters": np.asarray(process.starting_values(resids), dtype=float),
        "resids": resids,
        "sigma2": np.zeros(NOBS),
        "backcast": process.backcast(resids),
        "var_bounds": process.variance_bounds(resids),
    }
    return process, args


def mutations(process, args):
    """(argument, label, bad value, exception) for the arguments of compute_variance."""
    params, resids, bounds = args["parameters"], args["resids"], args["var_bounds"]
    rows = [
        ("parameters", "none", None, TypeError),
        ("parameters", "str", "abc", TypeError),
        ("parameters", "long", np.r_[params, 1.0], ValueError),
        ("resids", "none", None, TypeError),
        ("resids", "2d", np.ones((NOBS, 2)), ValueError),
        ("resids", "shorter", resids[:-1], ValueError),
        ("resids", "longer", np.r_[resids, resids[:5]], ValueError),
        ("resids", "empty", np.empty(0), ValueError),
        ("sigma2", "none", None, TypeError),
        ("sigma2", "list", [0.0] * NOBS, TypeError),
        ("sigma2", "tuple", (0.0,) * NOBS, TypeError),
        ("sigma2", "scalar", 0.0, TypeError),
        ("sigma2", "float32", np.zeros(NOBS, dtype=np.float32), TypeError),
        ("sigma2", "int64", np.zeros(NOBS, dtype=np.int64), TypeError),
        ("sigma2", "empty", np.empty(0), ValueError),
        ("sigma2", "short", np.zeros(NOBS - 1), ValueError),
        ("sigma2", "long", np.zeros(NOBS + 1), ValueError),
        ("sigma2", "2d", np.zeros((NOBS, 1)), ValueError),
        ("sigma2", "strided", np.zeros(2 * NOBS)[::2], ValueError),
        ("sigma2", "readonly", read_only(np.zeros(NOBS)), ValueError),
    ]
    if params.shape[0] > 0:
        rows += [
            ("parameters", "empty", np.empty(0), ValueError),
            ("parameters", "short", params[:-1], ValueError),
            ("parameters", "2d", np.ones((params.shape[0], 2)), ValueError),
        ]
    if process._uses_backcast:
        if process._backcast_size is None:
            rows += [
                ("backcast", "none", None, TypeError),
                ("backcast", "str", "abc", TypeError),
                ("backcast", "array", np.ones(2), TypeError),
                ("backcast", "empty", np.empty(0), TypeError),
            ]
        else:
            size = process._backcast_size
            rows += [
                ("backcast", "none", None, TypeError),
                ("backcast", "str", "abc", TypeError),
                ("backcast", "scalar", 1.0, ValueError),
                ("backcast", "short", np.ones(size - 1), ValueError),
                ("backcast", "empty", np.empty(0), ValueError),
            ]
    if process._uses_var_bounds:
        rows += [
            ("var_bounds", "none", None, TypeError),
            ("var_bounds", "short", bounds[:10], ValueError),
            ("var_bounds", "empty", np.empty((0, 2)), ValueError),
            ("var_bounds", "1d", bounds[:, 0], ValueError),
            ("var_bounds", "3cols", np.ones((NOBS, 3)), ValueError),
            ("var_bounds", "1col", np.ones((NOBS, 1)), ValueError),
        ]
    return rows


def _copy(value):
    return value.copy() if isinstance(value, np.ndarray) else value


def compute_variance_rows():
    rows = []
    for key in PROCESSES:
        process, args = setup_process(key)

        def call(*, process=process, **kwargs):
            return process.compute_variance(**kwargs)

        for argument, label, value, exc in mutations(process, args):
            kwargs = {k: _copy(v) for k, v in args.items()}
            kwargs[argument] = value
            if argument == "resids" and label in ("shorter", "longer", "empty"):
                # resids sets nobs, so it is var_bounds (if used) or sigma2 that differs
                name = "var_bounds" if process._uses_var_bounds else "sigma2"
            else:
                name = argument
            rows.append(
                Row(
                    key,
                    call,
                    kwargs,
                    exc,
                    rf"^{name} must be .+; got ",
                    id=f"{key}-{argument}-{label}",
                    name=name,
                )
            )
    return rows


@pytest.mark.parametrize("row", to_params(compute_variance_rows()))
def test_compute_variance_rejects_bad_arguments(row):
    run_row(row)


def conversion_cases():
    """(key, label, change, canonical): the same values in another form."""
    cases = []
    for key in PROCESSES:
        process, args = setup_process(key)
        resids, bounds = args["resids"], args["var_bounds"]
        ints = np.round(resids).astype(np.int64)
        small = np.minimum(bounds, 1e30)  # the largest float32 is 3e38
        cases += [
            (key, "parameters-list", {"parameters": args["parameters"].tolist()}, {}),
            (key, "resids-list", {"resids": resids.tolist()}, {}),
            (key, "resids-series", {"resids": pd.Series(resids)}, {}),
            (key, "resids-int", {"resids": ints}, {"resids": ints.astype(float)}),
            (key, "var_bounds-fortran", {"var_bounds": np.asfortranarray(bounds)}, {}),
            (key, "var_bounds-list", {"var_bounds": bounds.tolist()}, {}),
            (
                key,
                "var_bounds-float32",
                {"var_bounds": small.astype(np.float32)},
                {"var_bounds": small.astype(np.float32).astype(float)},
            ),
        ]
        if process._uses_backcast and process._backcast_size is None:
            cases += [(key, "backcast-int", {"backcast": 1}, {"backcast": 1.0})]
            cases += [
                (
                    key,
                    "backcast-float32",
                    {"backcast": np.float32(0.5)},
                    {"backcast": 0.5},
                )
            ]
        elif process._uses_backcast:
            as_list = {"backcast": list(args["backcast"])}
            cases += [(key, "backcast-list", as_list, {})]
    return cases


@pytest.mark.parametrize(
    ("key", "change", "canonical"),
    [
        pytest.param(key, change, canonical, id=f"{key}-{label}")
        for key, label, change, canonical in conversion_cases()
    ],
)
def test_compute_variance_converts_input(key, change, canonical):
    # The same values as a list, a Series, int64 or another memory layout give the
    # result of the float64 C contiguous arrays
    process, args = setup_process(key)
    expected_args = {k: _copy(v) for k, v in args.items()} | canonical
    expected = process.compute_variance(**expected_args)

    actual_args = {k: _copy(v) for k, v in args.items()} | change
    sigma2 = actual_args["sigma2"]
    actual = process.compute_variance(**actual_args)
    assert actual is sigma2
    assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("key", list(PROCESSES))
def test_compute_variance_valid_arguments_fill_sigma2_in_place(key):
    process, args = setup_process(key)
    sigma2 = args["sigma2"]
    out = process.compute_variance(**args)
    assert out is sigma2
    assert np.all(np.isfinite(sigma2))
    assert np.all(sigma2 > 0)


def test_garch_variance_matches_hand_recursion():
    # sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1], started at backcast
    process = GARCH(1, 0, 1)
    resids = np.random.default_rng(3).standard_normal(NOBS)
    omega, alpha, beta = 0.1, 0.1, 0.8
    backcast = 0.7
    expected = np.empty(NOBS)
    expected[0] = omega + alpha * backcast + beta * backcast
    for t in range(1, NOBS):
        expected[t] = omega + alpha * resids[t - 1] ** 2 + beta * expected[t - 1]
    bounds = np.column_stack([np.full(NOBS, 1e-8), np.full(NOBS, 1e8)])
    sigma2 = np.zeros(NOBS)
    out = process.compute_variance(
        [omega, alpha, beta], list(resids), sigma2, backcast, bounds.tolist()
    )
    assert_allclose(out, expected, rtol=1e-12)


def test_constant_and_fixed_variance_ignore_unused_arguments():
    process, args = setup_process("constant")
    out = process.compute_variance(
        args["parameters"], args["resids"], args["sigma2"], None, None
    )
    assert_allclose(out, args["parameters"][0])
    process, args = setup_process("fixed")
    out = process.compute_variance(
        args["parameters"], args["resids"], args["sigma2"], None, None
    )
    assert_allclose(out, np.linspace(0.5, 1.5, NOBS) * args["parameters"][0])
    process, args = setup_process("ewma-fixed")
    out = process.compute_variance(
        args["parameters"], args["resids"], args["sigma2"], args["backcast"], None
    )
    assert out.shape == (NOBS,)


def test_ewma_with_fixed_lambda_has_no_parameters():
    process, args = setup_process("ewma-fixed")
    assert args["parameters"].shape == (0,)
    process.compute_variance(**args)
    with pytest.raises(
        ValueError, match=r"^parameters must be .+ with shape \(0,\); got "
    ):
        process.compute_variance(**{**args, "parameters": np.array([0.94])})


def simulate_forecast_rows():
    rows = []
    for key in PROCESSES:
        process, args = setup_process(key)
        n = args["parameters"].shape[0]
        if n == 0 or key == "fixed":
            continue
        bad_parameters = [
            ("none", None, TypeError),
            ("str", "abc", TypeError),
            ("empty", np.empty(0), ValueError),
            ("short", args["parameters"][:-1], ValueError),
            ("long", np.r_[args["parameters"], 1.0], ValueError),
            ("2d", np.ones((n, 2)), ValueError),
        ]
        rng = np.random.default_rng(0).standard_normal

        def simulate(*, process=process, rng=rng, **kwargs):
            return process.simulate(kwargs["parameters"], 20, rng, burn=5)

        def forecast(*, process=process, **kwargs):
            return process.forecast(**kwargs)

        for label, value, exc in bad_parameters:
            match = r"^parameters must be .+; got "
            rows.append(
                Row(
                    key,
                    simulate,
                    {"parameters": value},
                    exc,
                    match,
                    id=f"simulate-{key}-parameters-{label}",
                    name="parameters",
                )
            )
            kwargs = {k: v for k, v in args.items() if k != "sigma2"}
            kwargs["parameters"] = value
            rows.append(
                Row(
                    key,
                    forecast,
                    kwargs,
                    exc,
                    match,
                    id=f"forecast-{key}-parameters-{label}",
                    name="parameters",
                )
            )
        # the other arrays of forecast
        kwargs = {k: v for k, v in args.items() if k != "sigma2"}
        extra = [
            ("resids", "none", None, TypeError),
            ("resids", "2d", np.ones((NOBS, 2)), ValueError),
        ]
        if process._uses_backcast:
            extra += [("backcast", "none", None, TypeError)]
        if process._uses_var_bounds:
            extra += [
                ("var_bounds", "none", None, TypeError),
                ("var_bounds", "short", args["var_bounds"][:10], ValueError),
                ("var_bounds", "1d", args["var_bounds"][:, 0], ValueError),
            ]
        for argument, label, value, exc in extra:
            rows.append(
                Row(
                    key,
                    forecast,
                    {**kwargs, argument: value},
                    exc,
                    rf"^{argument} must be .+; got ",
                    id=f"forecast-{key}-{argument}-{label}",
                    name=argument,
                )
            )
    return rows


@pytest.mark.parametrize("row", to_params(simulate_forecast_rows()))
def test_simulate_and_forecast_reject_bad_arguments(row):
    run_row(row)


@pytest.mark.parametrize("key", ["figarch", "fiaparch"])
def test_fractionally_integrated_processes_check_the_parameters_before_the_weights(key):
    # figarch_weights reads p + q + 1 values from the parameters it is given
    process, args = setup_process(key)
    short = args["parameters"][:2]
    rng = np.random.default_rng(1).standard_normal
    with pytest.raises(ValueError, match=r"^parameters must be .+; got shape \(2,\)"):
        process.simulate(short, 20, rng, burn=5)
    kwargs = {k: v for k, v in args.items() if k != "sigma2"}
    with pytest.raises(ValueError, match=r"^parameters must be .+; got shape \(2,\)"):
        process.forecast(**{**kwargs, "parameters": short})


def test_aparch_starting_values_with_common_asymmetry_have_the_parameter_length():
    # The candidates were evaluated with the unpacked parameters, which have a gamma
    # for each o, but compute_variance expects one gamma when common_asym is True
    resids = np.random.default_rng(5).standard_normal(300)
    for delta in (None, 1.5):
        process = APARCH(2, 2, 2, delta=delta, common_asym=True)
        sv = process.starting_values(resids)
        assert sv.shape == (process.num_params,)
        sigma2 = np.zeros(300)
        process.compute_variance(
            sv,
            resids,
            sigma2,
            process.backcast(resids),
            process.variance_bounds(resids),
        )
        assert np.all(np.isfinite(sigma2))
