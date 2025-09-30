from ._version import version as __version__, version_tuple
from .univariate.mean import arch_model
from .utility import test


def doc() -> None:
    import webbrowser  # noqa: PLC0415

    webbrowser.open("https://bashtage.github.io/arch/")


__all__ = ["__version__", "arch_model", "doc", "test", "version_tuple"]
