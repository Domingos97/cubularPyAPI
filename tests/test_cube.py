"""
Unit tests for the Cube class.
"""

import unittest
import numpy as np
from cubular.cube import Cube


class TestCube(unittest.TestCase):
    """Test cases for the Cube class."""
    
    def test_cube_initialization(self):
        """Test basic cube initialization."""
        cube = Cube(3)
        self.assertEqual(cube.size, 3)
        self.assertEqual(cube.data.shape, (3, 3, 3))
    
    def test_cube_initialization_with_data(self):
        """Test cube initialization with custom data."""
        data = np.ones((3, 3, 3))
        cube = Cube(3, data)
        self.assertTrue(np.array_equal(cube.data, data))
    
    def test_invalid_size(self):
        """Test that invalid sizes raise ValueError."""
        with self.assertRaises(ValueError):
            Cube(0)
        with self.assertRaises(ValueError):
            Cube(-1)
    
    def test_invalid_data_shape(self):
        """Test that mismatched data shape raises ValueError."""
        data = np.ones((2, 2, 2))
        with self.assertRaises(ValueError):
            Cube(3, data)
    
    def test_get_set(self):
        """Test getting and setting values."""
        cube = Cube(3)
        cube.set(0, 0, 0, 100)
        self.assertEqual(cube.get(0, 0, 0), 100)
    
    def test_rotate_x(self):
        """Test rotation around X axis."""
        cube = Cube(2)
        original_data = cube.data.copy()
        cube.rotate_x(1)
        self.assertFalse(np.array_equal(cube.data, original_data))
        
        # 4 rotations should return to original
        cube.rotate_x(3)
        self.assertTrue(np.array_equal(cube.data, original_data))
    
    def test_rotate_y(self):
        """Test rotation around Y axis."""
        cube = Cube(2)
        original_data = cube.data.copy()
        cube.rotate_y(1)
        self.assertFalse(np.array_equal(cube.data, original_data))
        
        # 4 rotations should return to original
        cube.rotate_y(3)
        self.assertTrue(np.array_equal(cube.data, original_data))
    
    def test_rotate_z(self):
        """Test rotation around Z axis."""
        cube = Cube(2)
        original_data = cube.data.copy()
        cube.rotate_z(1)
        self.assertFalse(np.array_equal(cube.data, original_data))
        
        # 4 rotations should return to original
        cube.rotate_z(3)
        self.assertTrue(np.array_equal(cube.data, original_data))
    
    def test_flip_x(self):
        """Test flipping along X axis."""
        cube = Cube(2)
        original_data = cube.data.copy()
        cube.flip_x()
        self.assertFalse(np.array_equal(cube.data, original_data))
        
        # Two flips should return to original
        cube.flip_x()
        self.assertTrue(np.array_equal(cube.data, original_data))
    
    def test_flip_y(self):
        """Test flipping along Y axis."""
        cube = Cube(2)
        original_data = cube.data.copy()
        cube.flip_y()
        self.assertFalse(np.array_equal(cube.data, original_data))
        
        # Two flips should return to original
        cube.flip_y()
        self.assertTrue(np.array_equal(cube.data, original_data))
    
    def test_flip_z(self):
        """Test flipping along Z axis."""
        cube = Cube(2)
        original_data = cube.data.copy()
        cube.flip_z()
        self.assertFalse(np.array_equal(cube.data, original_data))
        
        # Two flips should return to original
        cube.flip_z()
        self.assertTrue(np.array_equal(cube.data, original_data))
    
    def test_copy(self):
        """Test cube copying."""
        cube1 = Cube(3)
        cube2 = cube1.copy()
        
        self.assertEqual(cube1.size, cube2.size)
        self.assertTrue(np.array_equal(cube1.data, cube2.data))
        
        # Modifying copy shouldn't affect original
        cube2.set(0, 0, 0, 999)
        self.assertNotEqual(cube1.get(0, 0, 0), cube2.get(0, 0, 0))
    
    def test_equality(self):
        """Test cube equality."""
        cube1 = Cube(3)
        cube2 = Cube(3)
        self.assertEqual(cube1, cube2)
        
        cube2.set(0, 0, 0, 999)
        self.assertNotEqual(cube1, cube2)
    
    def test_repr(self):
        """Test string representation."""
        cube = Cube(3)
        self.assertIn("Cube(size=3)", repr(cube))
    
    def test_str(self):
        """Test detailed string representation."""
        cube = Cube(2)
        string = str(cube)
        self.assertIn("Cube(size=2)", string)


if __name__ == '__main__':
    unittest.main()
