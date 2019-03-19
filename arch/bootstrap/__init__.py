"""
Tools for implementing statistical bootstraps
"""
from __future__ import absolute_import

from arch.bootstrap import _samplers_python
from arch.bootstrap.base import (CircularBlockBootstrap, IIDBootstrap,
                                 IndependentSamplesBootstrap,
                                 MovingBlockBootstrap, StationaryBootstrap)
from arch.bootstrap.multiple_comparison import MCS, SPA, RealityCheck, StepM

COMPILED_SAMPLERS = True
try:
    from arch.bootstrap import _samplers
except ImportError:
    COMPILED_SAMPLERS = False


__all__ = ['IIDBootstrap', 'CircularBlockBootstrap', 'MovingBlockBootstrap',
           'StationaryBootstrap', 'IndependentSamplesBootstrap',
           'SPA', 'RealityCheck', 'StepM', 'MCS',
           '_samplers_python']

if COMPILED_SAMPLERS:
    __all__ += ['_samplers']
