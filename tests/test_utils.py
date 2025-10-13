"""
Unit tests for utility functions.
"""

import unittest
import numpy as np
from cubular.cube import Cube
from cubular.utils import (
    validate_size,
    create_identity_cube,
    create_empty_cube,
    create_filled_cube,
    get_slice,
    get_dimensions,
    is_symmetric
)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_validate_size_valid(self):
        """Test validation of valid sizes."""
        self.assertTrue(validate_size(1))
        self.assertTrue(validate_size(3))
        self.assertTrue(validate_size(100))
    
    def test_validate_size_invalid(self):
        """Test validation of invalid sizes."""
        self.assertFalse(validate_size(0))
        self.assertFalse(validate_size(-1))
        self.assertFalse(validate_size(3.5))
        self.assertFalse(validate_size("3"))
    
    def test_create_identity_cube(self):
        """Test creating identity cube."""
        cube = create_identity_cube(3)
        self.assertEqual(cube.size, 3)
        self.assertEqual(cube.data.shape, (3, 3, 3))
    
    def test_create_empty_cube(self):
        """Test creating empty cube."""
        cube = create_empty_cube(3)
        self.assertEqual(cube.size, 3)
        self.assertTrue(np.all(cube.data == 0))
    
    def test_create_filled_cube(self):
        """Test creating filled cube."""
        cube = create_filled_cube(3, 5)
        self.assertEqual(cube.size, 3)
        self.assertTrue(np.all(cube.data == 5))
    
    def test_get_slice_x(self):
        """Test getting slice along X axis."""
        cube = Cube(3)
        slice_data = get_slice(cube, 'x', 0)
        self.assertEqual(slice_data.shape, (3, 3))
        self.assertTrue(np.array_equal(slice_data, cube.data[0, :, :]))
    
    def test_get_slice_y(self):
        """Test getting slice along Y axis."""
        cube = Cube(3)
        slice_data = get_slice(cube, 'y', 1)
        self.assertEqual(slice_data.shape, (3, 3))
        self.assertTrue(np.array_equal(slice_data, cube.data[:, 1, :]))
    
    def test_get_slice_z(self):
        """Test getting slice along Z axis."""
        cube = Cube(3)
        slice_data = get_slice(cube, 'z', 2)
        self.assertEqual(slice_data.shape, (3, 3))
        self.assertTrue(np.array_equal(slice_data, cube.data[:, :, 2]))
    
    def test_get_slice_invalid_axis(self):
        """Test that invalid axis raises ValueError."""
        cube = Cube(3)
        with self.assertRaises(ValueError):
            get_slice(cube, 'w', 0)
    
    def test_get_slice_out_of_bounds(self):
        """Test that out of bounds index raises ValueError."""
        cube = Cube(3)
        with self.assertRaises(ValueError):
            get_slice(cube, 'x', 3)
        with self.assertRaises(ValueError):
            get_slice(cube, 'x', -1)
    
    def test_get_dimensions(self):
        """Test getting cube dimensions."""
        cube = Cube(3)
        dims = get_dimensions(cube)
        self.assertEqual(dims, (3, 3, 3))
        
        cube2 = Cube(5)
        dims2 = get_dimensions(cube2)
        self.assertEqual(dims2, (5, 5, 5))
    
    def test_is_symmetric_x(self):
        """Test symmetry check along X axis."""
        # Create a symmetric cube
        data = np.array([
            [[1, 2, 1], [3, 4, 3], [1, 2, 1]],
            [[3, 4, 3], [5, 6, 5], [3, 4, 3]],
            [[1, 2, 1], [3, 4, 3], [1, 2, 1]]
        ])
        cube = Cube(3, data)
        self.assertTrue(is_symmetric(cube, 'x'))
    
    def test_is_not_symmetric(self):
        """Test asymmetric cube."""
        cube = Cube(3)
        # Default sequential cube is not symmetric
        self.assertFalse(is_symmetric(cube, 'x'))
    
    def test_is_symmetric_invalid_axis(self):
        """Test that invalid axis raises ValueError."""
        cube = Cube(3)
        with self.assertRaises(ValueError):
            is_symmetric(cube, 'w')


if __name__ == '__main__':
    unittest.main()
