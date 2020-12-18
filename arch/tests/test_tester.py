import pytest

import arch


def test_runner():
    status = arch.test(location="tests/utility/test_cov.py", exit=False)
    assert status == 0


def test_runner_exception():
    with pytest.raises(RuntimeError):
        arch.test(location="tests/utility/unknown_location.py")


def test_extra_args():
    status = arch.test(
        "--tb=short",
        append=False,
        location="tests/utility/test_cov.py",
        exit=False,
    )

    assert status == 0

    status = arch.test(
        ["-r", "a"],
        append=True,
        location="tests/utility/test_cov.py",
        exit=False,
    )

    assert status == 0
