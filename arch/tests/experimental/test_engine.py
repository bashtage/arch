import pytest

from arch.experimental.engine import (
    _SUPPORTED_ENGINES,
    linalg,
    numpy,
    use_backend,
    set_backend,
    backend,
)


def test_numpy_name():
    for engine_name in _SUPPORTED_ENGINES:
        if engine_name == "tensorflow":
            continue

        with use_backend(engine_name):
            assert engine_name == numpy.name


def test_linalg_name():
    for engine_name in _SUPPORTED_ENGINES:
        if engine_name == "tensorflow":
            continue

        with use_backend(engine_name):
            assert engine_name == linalg.name


def test_set_backend_eq():
    assert set_backend(backend()) == backend()


def test_numpy_getattribute():
    import numpy as np

    with use_backend("numpy"):
        array = numpy.array
        assert array == np.array


def test_linalg_getattribute():
    import numpy.linalg

    with use_backend("numpy"):
        inv = linalg.inv
        assert inv == numpy.linalg.inv


def test_numpy_getattribute_failed_attr():
    with use_backend("numpy"):
        with pytest.raises(AttributeError) as exc:
            numpy.xyz  # noqa

    assert str(exc.value) == (
        "Cannot get attribute / function (xyz) from numpy library in "
        "backend engine numpy"
    )


def test_linalg_getattribute_failed_attr():
    with use_backend("numpy"):
        with pytest.raises(AttributeError) as exc:
            linalg.xyz  # noqa

    assert str(exc.value) == (
        "Cannot get attribute / function (xyz) from linalg library in "
        "backend engine numpy"
    )


def test_numpy_getattribute_failed_import():
    with use_backend("jax"):
        with pytest.raises(ImportError) as exc:
            numpy.array  # noqa

    assert str(exc.value) == (
        "Library `numpy` cannot be imported from backend engine "
        "jax. Please make sure to install the library "
        "via `pip install jax`."
    )


def test_linalg_getattribute_failed_attr():
    with use_backend("jax"):
        with pytest.raises(ImportError) as exc:
            linalg.array  # noqa

    assert str(exc.value) == (
        "Library `linalg` cannot be imported from backend engine "
        "jax. Please make sure to install the library "
        "via `pip install jax`."
    )
