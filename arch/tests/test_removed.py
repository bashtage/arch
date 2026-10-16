import importlib

import pytest

import arch.utility.array
import arch.utility.exceptions


@pytest.mark.parametrize("name", ["arch.__future__", "arch.__future__.reindexing"])
def test_future_package_removed(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


@pytest.mark.parametrize(
    "module", [arch.utility.array, arch.utility.exceptions], ids=["array", "exceptions"]
)
def test_deprecation_doc_removed(module):
    assert not hasattr(module, "deprecation_doc")
