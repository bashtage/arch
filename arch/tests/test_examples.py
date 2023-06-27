import glob
import os
import sys

import pytest

SKIP = True
REASON = "Required packages not available"

try:
    import jupyter_client

    # matplotlib is required for most notebooks
    import matplotlib  # noqa: F401
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat

    kernels = jupyter_client.kernelspec.find_kernel_specs()
    SKIP = False

    if sys.platform.startswith("win") and sys.version_info >= (
        3,
        8,
    ):  # pragma: no cover
        import asyncio

        try:
            from asyncio import WindowsSelectorEventLoopPolicy
        except ImportError:
            pass  # Can't assign a policy which doesn't exist.
        else:
            if not isinstance(
                asyncio.get_event_loop_policy(), WindowsSelectorEventLoopPolicy
            ):
                asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

except ImportError:  # pragma: no cover
    pytestmark = pytest.mark.skip(reason=REASON)

SLOW_NOTEBOOKS = ["multiple-comparison_examples.ipynb"]
if bool(os.environ.get("ARCH_TEST_SLOW_NOTEBOOKS", False)):  # pragma: no cover
    SLOW_NOTEBOOKS = []
kernel_name = "python%s" % sys.version_info.major

head, _ = os.path.split(__file__)
NOTEBOOK_DIR = os.path.abspath(os.path.join(head, "..", "..", "examples"))

nbs = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, "*.ipynb")))
ids = [os.path.split(nb)[-1].split(".")[0] for nb in nbs]
if not nbs:  # pragma: no cover
    REASON = "No notebooks found and so no tests run"
    pytestmark = pytest.mark.skip(reason=REASON)


@pytest.mark.slow
@pytest.mark.parametrize("notebook", nbs, ids=ids)
@pytest.mark.skipif(SKIP, reason=REASON)
def test_notebook(notebook):
    nb_name = os.path.split(notebook)[-1]
    if nb_name in SLOW_NOTEBOOKS:
        pytest.skip("Notebook is too slow to test")
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(allow_errors=False, timeout=240, kernel_name=kernel_name)
    ep.preprocess(nb, {"metadata": {"path": NOTEBOOK_DIR}})
