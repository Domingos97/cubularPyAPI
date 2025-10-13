"""
Utility functions for the Cubular API.
"""

import numpy as np
from typing import Tuple, Optional
from .cube import Cube


def validate_size(size: int) -> bool:
    """
    Validate that a cube size is acceptable.
    
    Args:
        size (int): The size to validate
        
    Returns:
        bool: True if size is valid, False otherwise
    """
    return isinstance(size, int) and size >= 1


def create_identity_cube(size: int) -> Cube:
    """
    Create a cube with sequential values.
    
    Args:
        size (int): Size of the cube
        
    Returns:
        Cube: A new cube with sequential values
    """
    return Cube(size)


def create_empty_cube(size: int) -> Cube:
    """
    Create a cube initialized with zeros.
    
    Args:
        size (int): Size of the cube
        
    Returns:
        Cube: A new cube with all zeros
    """
    data = np.zeros((size, size, size), dtype=int)
    return Cube(size, data)


def create_filled_cube(size: int, value: int = 1) -> Cube:
    """
    Create a cube filled with a specific value.
    
    Args:
        size (int): Size of the cube
        value (int): Value to fill the cube with (default: 1)
        
    Returns:
        Cube: A new cube filled with the specified value
    """
    data = np.full((size, size, size), value, dtype=int)
    return Cube(size, data)


def get_slice(cube: Cube, axis: str, index: int) -> np.ndarray:
    """
    Get a 2D slice of the cube along a specified axis.
    
    Args:
        cube (Cube): The cube to slice
        axis (str): The axis to slice along ('x', 'y', or 'z')
        index (int): The index of the slice
        
    Returns:
        np.ndarray: A 2D array representing the slice
        
    Raises:
        ValueError: If axis is invalid or index is out of bounds
    """
    axis = axis.lower()
    
    if index < 0 or index >= cube.size:
        raise ValueError(f"Index {index} is out of bounds for cube size {cube.size}")
    
    if axis == 'x':
        return cube.data[index, :, :]
    elif axis == 'y':
        return cube.data[:, index, :]
    elif axis == 'z':
        return cube.data[:, :, index]
    else:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")


def get_dimensions(cube: Cube) -> Tuple[int, int, int]:
    """
    Get the dimensions of a cube.
    
    Args:
        cube (Cube): The cube to get dimensions from
        
    Returns:
        Tuple[int, int, int]: The dimensions (size, size, size)
    """
    return (cube.size, cube.size, cube.size)


def is_symmetric(cube: Cube, axis: str) -> bool:
    """
    Check if a cube is symmetric along a specified axis.
    
    Args:
        cube (Cube): The cube to check
        axis (str): The axis to check symmetry along ('x', 'y', or 'z')
        
    Returns:
        bool: True if the cube is symmetric, False otherwise
    """
    axis = axis.lower()
    
    if axis == 'x':
        flipped = np.flip(cube.data, axis=0)
    elif axis == 'y':
        flipped = np.flip(cube.data, axis=1)
    elif axis == 'z':
        flipped = np.flip(cube.data, axis=2)
    else:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")
    
    return np.array_equal(cube.data, flipped)
