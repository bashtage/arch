import tempfile

import pytest

from arch import __version__, version_tuple

try:
    from arch._build.git_version import get_version, write_version_file

    HAS_SETUPTOOLS_SCM = True
except ImportError:
    HAS_SETUPTOOLS_SCM = False


@pytest.mark.skipif(not HAS_SETUPTOOLS_SCM, reason="setuptools_scm is not installed")
def test_get_version():
    try:
        version, version_fields = get_version()

        assert isinstance(version, str)
        assert isinstance(version_fields, tuple)
        assert all(isinstance(v, (int, str)) for v in version_fields)
    except LookupError:
        pytest.skip("No git repository found")


@pytest.mark.skipif(not HAS_SETUPTOOLS_SCM, reason="setuptools_scm is not installed")
def test_write_version_file():
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:

        write_version_file(tmpfile.name, __version__, version_tuple)
        with open(tmpfile.name, "r") as f:
            content = f.read()

            assert f"__version__ = version = '{__version__}'" in content
            assert f"__version_tuple__ = version_tuple = {version_tuple}" in content
