import logging
import os

import pandas as pd
import pytest

pytest_plugins = [
    "arch.tests.unitroot.cointegration_data",
]


logger = logging.getLogger(__name__)
COW = bool(os.environ.get("ARCH_TEST_COPY_ON_WRITE", ""))
try:
    pd.options.mode.copy_on_write = COW
except AttributeError:
    pass

if COW:
    logger.critical("Copy on Write Enabled!")
else:
    logger.critical("Copy on Write disabled")


def pytest_configure(config):
    # Minimal config to simplify running tests from lm.test()
    config.addinivalue_line("markers", "slow: mark a test as slow")
    config.addinivalue_line(
        "filterwarnings", "ignore:Method .ptp is deprecated:FutureWarning"
    )


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true", help="skip slow tests")
    parser.addoption("--only-slow", action="store_true", help="run only slow tests")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and item.config.getoption(
        "--skip-slow"
    ):  # pragma: no cover
        pytest.skip("skipping due to --skip-slow")  # pragma: no cover

    if "slow" not in item.keywords and item.config.getoption(
        "--only-slow"
    ):  # pragma: no cover
        pytest.skip("skipping due to --only-slow")  # pragma: no cover


@pytest.fixture
def agg_backend():
    """
    Fixture that switches the backend to agg for the duration of the test

    Returns
    -------
    switch_backend : callable
        Function that will change the backend to agg when called

    Notes
    -----
    Used by passing as an argument to the function that produces a plot,
    for example

    def test_some_plot(agg_backend):
        <test code>
    """
    backend = None
    try:
        import matplotlib as mpl  # noqa: PLC0415

        backend = mpl.get_backend()
        mpl.use("agg")

    except ImportError:
        # Nothing to do if MPL is not available
        pass

    def null():
        pass

    yield null
    if backend:
        import matplotlib as mpl  # noqa: PLC0415

        mpl.use(backend)
