from ._version import version as __version__, version_tuple
from .univariate.mean import arch_model
from .utility import test


def doc() -> None:
    import webbrowser

    webbrowser.open("https://bashtage.github.io/arch/")


__all__ = ["arch_model", "__version__", "doc", "test", "version_tuple"]
