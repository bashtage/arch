try:
    import matplotlib.pyplot as plt  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

__all__ = ["HAS_MATPLOTLIB"]
