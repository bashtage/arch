"""
Tools for implementing statistical bootstraps
"""

from arch.bootstrap import _samplers_python
from arch.bootstrap.base import (
    CircularBlockBootstrap,
    IIDBootstrap,
    IndependentSamplesBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
    optimal_block_length,
)
from arch.bootstrap.multiple_comparison import MCS, SPA, RealityCheck, StepM

COMPILED_SAMPLERS = True
try:
    from arch.bootstrap import _samplers
except ImportError:
    COMPILED_SAMPLERS = False


__all__ = [
    "IIDBootstrap",
    "CircularBlockBootstrap",
    "MovingBlockBootstrap",
    "StationaryBootstrap",
    "IndependentSamplesBootstrap",
    "SPA",
    "RealityCheck",
    "StepM",
    "MCS",
    "_samplers_python",
    "optimal_block_length",
]

if COMPILED_SAMPLERS:
    __all__ += ["_samplers"]
