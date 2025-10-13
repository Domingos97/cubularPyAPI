"""
Operations that can be performed on cubes.
"""

import numpy as np
from typing import Tuple
from .cube import Cube


def rotate(cube: Cube, axis: str, k: int = 1) -> Cube:
    """
    Rotate a cube around a specified axis.
    
    Args:
        cube (Cube): The cube to rotate
        axis (str): The axis to rotate around ('x', 'y', or 'z')
        k (int): Number of 90-degree rotations (default: 1)
    
    Returns:
        Cube: A new rotated cube
        
    Raises:
        ValueError: If axis is not 'x', 'y', or 'z'
    """
    result = cube.copy()
    axis = axis.lower()
    
    if axis == 'x':
        result.rotate_x(k)
    elif axis == 'y':
        result.rotate_y(k)
    elif axis == 'z':
        result.rotate_z(k)
    else:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")
    
    return result


def flip(cube: Cube, axis: str) -> Cube:
    """
    Flip a cube along a specified axis.
    
    Args:
        cube (Cube): The cube to flip
        axis (str): The axis to flip along ('x', 'y', or 'z')
    
    Returns:
        Cube: A new flipped cube
        
    Raises:
        ValueError: If axis is not 'x', 'y', or 'z'
    """
    result = cube.copy()
    axis = axis.lower()
    
    if axis == 'x':
        result.flip_x()
    elif axis == 'y':
        result.flip_y()
    elif axis == 'z':
        result.flip_z()
    else:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")
    
    return result


def transform(cube: Cube, operations: list) -> Cube:
    """
    Apply a sequence of transformations to a cube.
    
    Args:
        cube (Cube): The cube to transform
        operations (list): List of tuples (operation, axis, *args)
                          e.g., [('rotate', 'x', 1), ('flip', 'y')]
    
    Returns:
        Cube: The transformed cube
        
    Example:
        >>> c = Cube(3)
        >>> ops = [('rotate', 'x', 1), ('flip', 'z')]
        >>> result = transform(c, ops)
    """
    result = cube.copy()
    
    for op in operations:
        if len(op) < 2:
            raise ValueError("Each operation must have at least (operation_type, axis)")
        
        op_type = op[0].lower()
        axis = op[1]
        
        if op_type == 'rotate':
            k = op[2] if len(op) > 2 else 1
            result = rotate(result, axis, k)
        elif op_type == 'flip':
            result = flip(result, axis)
        else:
            raise ValueError(f"Unknown operation type '{op_type}'. Must be 'rotate' or 'flip'")
    
    return result


def merge(cube1: Cube, cube2: Cube, operation: str = 'add') -> Cube:
    """
    Merge two cubes element-wise.
    
    Args:
        cube1 (Cube): First cube
        cube2 (Cube): Second cube
        operation (str): Operation to perform ('add', 'subtract', 'multiply', 'max', 'min')
    
    Returns:
        Cube: Result of the merge operation
        
    Raises:
        ValueError: If cubes have different sizes or operation is invalid
    """
    if cube1.size != cube2.size:
        raise ValueError(f"Cubes must have the same size. Got {cube1.size} and {cube2.size}")
    
    operation = operation.lower()
    
    if operation == 'add':
        result_data = cube1.data + cube2.data
    elif operation == 'subtract':
        result_data = cube1.data - cube2.data
    elif operation == 'multiply':
        result_data = cube1.data * cube2.data
    elif operation == 'max':
        result_data = np.maximum(cube1.data, cube2.data)
    elif operation == 'min':
        result_data = np.minimum(cube1.data, cube2.data)
    else:
        raise ValueError(f"Unknown operation '{operation}'. Must be 'add', 'subtract', 'multiply', 'max', or 'min'")
    
    return Cube(cube1.size, result_data)
