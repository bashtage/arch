import glob
import os
import sys

import pytest

SKIP = True

try:
    import jupyter_client

    # matplotlib is required for most notebooks
    import matplotlib  # noqa: F401
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat

    kernels = jupyter_client.kernelspec.find_kernel_specs()
    SKIP = False
except ImportError:  # pragma: no cover
    pytestmark = pytest.mark.skip(reason="Required packages not available")

SLOW_NOTEBOOKS = ["multiple-comparison_examples.ipynb"]
if bool(os.environ.get("ARCH_TEST_SLOW_NOTEBOOKS", False)):
    SLOW_NOTEBOOKS = []
kernel_name = "python%s" % sys.version_info.major

head, _ = os.path.split(__file__)
NOTEBOOK_DIR = os.path.abspath(os.path.join(head, "..", "..", "examples"))

nbs = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, "*.ipynb")))
ids = list(map(lambda s: os.path.split(s)[-1].split(".")[0], nbs))
if not nbs:  # pragma: no cover
    pytest.mark.skip(reason="No notebooks found and so no tests run")


@pytest.fixture(params=nbs, ids=ids)
def notebook(request):
    return request.param


@pytest.mark.slow
@pytest.mark.skipif(SKIP, reason="Required packages not available")
def test_notebook(notebook):
    nb_name = os.path.split(notebook)[-1]
    if nb_name in SLOW_NOTEBOOKS:
        pytest.skip("Notebook is too slow to test")
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(allow_errors=False, timeout=240, kernel_name=kernel_name)
    ep.preprocess(nb, {"metadata": {"path": NOTEBOOK_DIR}})
