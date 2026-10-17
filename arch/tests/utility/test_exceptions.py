import numpy as np
import pytest

from arch.unitroot import ADF, KPSS, PhillipsPerron, VarianceRatio
from arch.utility.exceptions import InfeasibleTestException


def test_infeasible_test_exception_is_value_error_and_runtime_error():
    assert issubclass(InfeasibleTestException, ValueError)
    assert issubclass(InfeasibleTestException, RuntimeError)
    assert issubclass(InfeasibleTestException, Exception)


@pytest.mark.parametrize("caught", [RuntimeError, ValueError])
def test_raised_exception_is_caught_by_both(caught):
    found = None
    try:
        raise InfeasibleTestException("not feasible")
    except caught as exc:
        found = exc
    assert isinstance(found, InfeasibleTestException)
    assert str(found) == "not feasible"


@pytest.mark.parametrize("caught", [RuntimeError, ValueError], ids=["runtime", "value"])
@pytest.mark.parametrize(
    "test",
    [ADF, KPSS, PhillipsPerron, VarianceRatio],
    ids=["ADF", "KPSS", "PhillipsPerron", "VarianceRatio"],
)
def test_short_series_raises_an_exception_that_both_catch(test, caught):
    # A series that is too short for the test
    y = np.array([0.0, 1.0])
    with pytest.raises(caught) as info:
        test(y).stat  # noqa: B018
    assert isinstance(info.value, InfeasibleTestException)
