"""
Core Cube class for representing and manipulating 3D cubes.
"""

import numpy as np
from typing import Tuple, Optional


class Cube:
    """
    Represents a 3D cube with specified dimensions.
    
    Attributes:
        size (int): The size of the cube (n x n x n)
        data (np.ndarray): The 3D array representing the cube data
    """
    
    def __init__(self, size: int = 3, data: Optional[np.ndarray] = None):
        """
        Initialize a Cube.
        
        Args:
            size (int): Size of the cube (default: 3)
            data (np.ndarray, optional): Initial data for the cube
        
        Raises:
            ValueError: If size is less than 1
        """
        if size < 1:
            raise ValueError("Cube size must be at least 1")
        
        self.size = size
        
        if data is not None:
            if data.shape != (size, size, size):
                raise ValueError(f"Data shape {data.shape} doesn't match cube size ({size}, {size}, {size})")
            self.data = data.copy()
        else:
            # Initialize with sequential values for easy tracking
            self.data = np.arange(size ** 3).reshape((size, size, size))
    
    def get(self, x: int, y: int, z: int) -> int:
        """
        Get value at specific coordinates.
        
        Args:
            x, y, z: Coordinates in the cube
            
        Returns:
            Value at the specified position
        """
        return self.data[x, y, z]
    
    def set(self, x: int, y: int, z: int, value: int):
        """
        Set value at specific coordinates.
        
        Args:
            x, y, z: Coordinates in the cube
            value: Value to set
        """
        self.data[x, y, z] = value
    
    def rotate_x(self, k: int = 1):
        """
        Rotate cube around X axis.
        
        Args:
            k (int): Number of 90-degree rotations (default: 1)
        """
        k = k % 4  # Normalize rotation
        self.data = np.rot90(self.data, k=k, axes=(1, 2))
    
    def rotate_y(self, k: int = 1):
        """
        Rotate cube around Y axis.
        
        Args:
            k (int): Number of 90-degree rotations (default: 1)
        """
        k = k % 4
        self.data = np.rot90(self.data, k=k, axes=(0, 2))
    
    def rotate_z(self, k: int = 1):
        """
        Rotate cube around Z axis.
        
        Args:
            k (int): Number of 90-degree rotations (default: 1)
        """
        k = k % 4
        self.data = np.rot90(self.data, k=k, axes=(0, 1))
    
    def flip_x(self):
        """Flip cube along X axis."""
        self.data = np.flip(self.data, axis=0)
    
    def flip_y(self):
        """Flip cube along Y axis."""
        self.data = np.flip(self.data, axis=1)
    
    def flip_z(self):
        """Flip cube along Z axis."""
        self.data = np.flip(self.data, axis=2)
    
    def copy(self) -> 'Cube':
        """
        Create a deep copy of the cube.
        
        Returns:
            A new Cube instance with copied data
        """
        return Cube(self.size, self.data)
    
    def __repr__(self) -> str:
        """String representation of the cube."""
        return f"Cube(size={self.size})"
    
    def __str__(self) -> str:
        """Detailed string representation."""
        return f"Cube(size={self.size})\n{self.data}"
    
    def __eq__(self, other: 'Cube') -> bool:
        """Check equality with another cube."""
        if not isinstance(other, Cube):
            return False
        return self.size == other.size and np.array_equal(self.data, other.data)
