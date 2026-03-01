"""
Compute backends for Variational Encrypted MPC.

PlaintextBackend is available by default.
But OpenFHEBackend requires OpenFHE Python bindings.
"""

from .base import BackendBase
from .plaintext_backend import PlaintextBackend

__all__ = ["BackendBase", "PlaintextBackend"]

try:
    from .openfhe_backend import OpenFHEBackend
    __all__.append("OpenFHEBackend")
except ImportError:
    pass