import pytest

from arch import utility


@pytest.mark.slow
@pytest.mark.parametrize("arg", [["--collect-only"], "--collect-only"])
def test_runner(arg):
    utility.test(arg, exit=False)
