import importlib

import pytest


@pytest.mark.parametrize("name", ["arch.__future__", "arch.__future__.reindexing"])
def test_future_package_removed(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)
