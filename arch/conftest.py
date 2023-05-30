import logging
import os

import pandas as pd
import pytest

pytest_plugins = [
    "arch.tests.unitroot.cointegration_data",
]


logger = logging.getLogger(__name__)
COW = bool(os.environ.get("ARCH_TEST_COPY_ON_WRITE", False))
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
