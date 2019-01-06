import pytest


def pytest_addoption(parser):
    parser.addoption("--skip-slow", action="store_true",
                     help="skip slow tests")
    parser.addoption("--only-slow", action="store_true",
                     help="run only slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and item.config.getoption("--skip-slow"):  # pragma: no cover
        pytest.skip("skipping due to --skip-slow")  # pragma: no cover

    if 'slow' not in item.keywords and item.config.getoption("--only-slow"):  # pragma: no cover
        pytest.skip("skipping due to --only-slow")  # pragma: no cover
