import importlib.util
import os
from pathlib import Path

import pytest

import arch

GIT_VERSION = Path(arch.__file__).parent / "_build" / "git_version.py"


@pytest.fixture(scope="module")
def git_version():
    pytest.importorskip("setuptools_scm")
    if not GIT_VERSION.exists():
        pytest.skip("The build scripts are not installed")
    spec = importlib.util.spec_from_file_location("arch_git_version", GIT_VERSION)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_version_file(git_version, tmp_path):
    # The editable install rewrites the version file whenever a process imports
    # arch, while other processes may import it at the same time. The file must
    # never be truncated: an unchanged file is not written again and a changed one
    # is replaced in one step.
    target = tmp_path / "_version.py"
    git_version.write_version_file(str(target), "1.2.3", (1, 2, 3))
    assert "__version__ = version = '1.2.3'" in target.read_text()

    old = 1_000_000_000
    os.utime(target, (old, old))
    git_version.write_version_file(str(target), "1.2.3", (1, 2, 3))
    assert target.stat().st_mtime == old

    file_id = target.stat().st_ino
    git_version.write_version_file(str(target), "1.2.4", (1, 2, 4))
    assert "__version__ = version = '1.2.4'" in target.read_text()
    assert target.stat().st_ino != file_id
    assert [p.name for p in tmp_path.iterdir()] == ["_version.py"]


def test_write_version_file_retries_replace(git_version, tmp_path, monkeypatch):
    # Windows cannot replace a file that another process has open, which an
    # import does briefly, and so the replacement is retried
    target = tmp_path / "_version.py"
    git_version.write_version_file(str(target), "1.2.3", (1, 2, 3))
    replace = Path.replace
    calls = []

    def busy_twice(self, other):
        calls.append(other)
        if len(calls) <= 2:
            raise PermissionError("the file is open")
        return replace(self, other)

    monkeypatch.setattr(git_version.time, "sleep", lambda _: None)
    monkeypatch.setattr(Path, "replace", busy_twice)
    git_version.write_version_file(str(target), "1.2.4", (1, 2, 4))
    assert len(calls) == 3
    assert "__version__ = version = '1.2.4'" in target.read_text()
    assert [p.name for p in tmp_path.iterdir()] == ["_version.py"]


def test_write_version_file_gives_up(git_version, tmp_path, monkeypatch):
    target = tmp_path / "_version.py"
    git_version.write_version_file(str(target), "1.2.3", (1, 2, 3))

    def always_busy(self, other):
        raise PermissionError("the file is open")

    monkeypatch.setattr(git_version.time, "sleep", lambda _: None)
    monkeypatch.setattr(Path, "replace", always_busy)
    with pytest.raises(PermissionError):
        git_version.write_version_file(str(target), "1.2.4", (1, 2, 4))
    # The file is unchanged, and the temporary file is removed
    assert "__version__ = version = '1.2.3'" in target.read_text()
    assert [p.name for p in tmp_path.iterdir()] == ["_version.py"]
