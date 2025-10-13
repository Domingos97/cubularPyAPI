"""
Cubular Python API
A Python library for cube manipulation and operations.
"""

__version__ = "0.1.0"
__author__ = "Domingos97"

from .cube import Cube
from .operations import rotate, flip, transform
from .utils import validate_size, create_identity_cube

__all__ = [
    "Cube",
    "rotate",
    "flip",
    "transform",
    "validate_size",
    "create_identity_cube",
]
