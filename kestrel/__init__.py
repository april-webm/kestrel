# kestrel/__init__.py
"""
Kestrel: A Modern Stochastic Modelling Library.

Provides unified interface for fitting and simulating stochastic processes.
"""

__version__ = "0.1.0"

from kestrel.base import StochasticProcess
from kestrel.diffusion import (
    BrownianMotion,
    GeometricBrownianMotion,
    OUProcess,
    CIRProcess,
)
from kestrel.jump_diffusion import MertonProcess
from kestrel.utils import KestrelResult

__all__ = [
    # Base
    "StochasticProcess",
    # Diffusion processes
    "BrownianMotion",
    "GeometricBrownianMotion",
    "OUProcess",
    "CIRProcess",
    # Jump diffusion processes
    "MertonProcess",
    # Utilities
    "KestrelResult",
]
