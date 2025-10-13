"""
Unit tests for cube operations.
"""

import unittest
import numpy as np
from cubular.cube import Cube
from cubular.operations import rotate, flip, transform, merge


class TestOperations(unittest.TestCase):
    """Test cases for cube operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cube = Cube(3)
    
    def test_rotate_x(self):
        """Test rotate function with X axis."""
        result = rotate(self.cube, 'x', 1)
        self.assertIsInstance(result, Cube)
        self.assertNotEqual(result, self.cube)
    
    def test_rotate_y(self):
        """Test rotate function with Y axis."""
        result = rotate(self.cube, 'y', 1)
        self.assertIsInstance(result, Cube)
        self.assertNotEqual(result, self.cube)
    
    def test_rotate_z(self):
        """Test rotate function with Z axis."""
        result = rotate(self.cube, 'z', 1)
        self.assertIsInstance(result, Cube)
        self.assertNotEqual(result, self.cube)
    
    def test_rotate_invalid_axis(self):
        """Test that invalid axis raises ValueError."""
        with self.assertRaises(ValueError):
            rotate(self.cube, 'w', 1)
    
    def test_rotate_multiple(self):
        """Test multiple rotations."""
        result = rotate(self.cube, 'x', 4)
        # 4 rotations of 90 degrees should return to original
        self.assertEqual(result, self.cube)
    
    def test_flip_x(self):
        """Test flip function with X axis."""
        result = flip(self.cube, 'x')
        self.assertIsInstance(result, Cube)
        
        # Double flip should return to original
        result2 = flip(result, 'x')
        self.assertEqual(result2, self.cube)
    
    def test_flip_y(self):
        """Test flip function with Y axis."""
        result = flip(self.cube, 'y')
        self.assertIsInstance(result, Cube)
        
        # Double flip should return to original
        result2 = flip(result, 'y')
        self.assertEqual(result2, self.cube)
    
    def test_flip_z(self):
        """Test flip function with Z axis."""
        result = flip(self.cube, 'z')
        self.assertIsInstance(result, Cube)
        
        # Double flip should return to original
        result2 = flip(result, 'z')
        self.assertEqual(result2, self.cube)
    
    def test_flip_invalid_axis(self):
        """Test that invalid axis raises ValueError."""
        with self.assertRaises(ValueError):
            flip(self.cube, 'w')
    
    def test_transform_single_operation(self):
        """Test transform with single operation."""
        ops = [('rotate', 'x', 1)]
        result = transform(self.cube, ops)
        expected = rotate(self.cube, 'x', 1)
        self.assertEqual(result, expected)
    
    def test_transform_multiple_operations(self):
        """Test transform with multiple operations."""
        ops = [('rotate', 'x', 1), ('flip', 'y')]
        result = transform(self.cube, ops)
        self.assertIsInstance(result, Cube)
    
    def test_transform_invalid_operation(self):
        """Test that invalid operation raises ValueError."""
        ops = [('invalid', 'x')]
        with self.assertRaises(ValueError):
            transform(self.cube, ops)
    
    def test_merge_add(self):
        """Test merge with add operation."""
        cube1 = Cube(3)
        cube2 = Cube(3)
        result = merge(cube1, cube2, 'add')
        expected_data = cube1.data + cube2.data
        self.assertTrue(np.array_equal(result.data, expected_data))
    
    def test_merge_subtract(self):
        """Test merge with subtract operation."""
        cube1 = Cube(3)
        cube2 = Cube(3)
        result = merge(cube1, cube2, 'subtract')
        expected_data = cube1.data - cube2.data
        self.assertTrue(np.array_equal(result.data, expected_data))
    
    def test_merge_multiply(self):
        """Test merge with multiply operation."""
        cube1 = Cube(3)
        cube2 = Cube(3)
        result = merge(cube1, cube2, 'multiply')
        expected_data = cube1.data * cube2.data
        self.assertTrue(np.array_equal(result.data, expected_data))
    
    def test_merge_max(self):
        """Test merge with max operation."""
        cube1 = Cube(3)
        cube2 = Cube(3)
        cube2.set(0, 0, 0, 1000)
        result = merge(cube1, cube2, 'max')
        self.assertEqual(result.get(0, 0, 0), 1000)
    
    def test_merge_min(self):
        """Test merge with min operation."""
        cube1 = Cube(3)
        cube2 = Cube(3)
        cube1.set(0, 0, 0, 1000)
        result = merge(cube1, cube2, 'min')
        self.assertEqual(result.get(0, 0, 0), cube2.get(0, 0, 0))
    
    def test_merge_different_sizes(self):
        """Test that merging different sized cubes raises ValueError."""
        cube1 = Cube(3)
        cube2 = Cube(4)
        with self.assertRaises(ValueError):
            merge(cube1, cube2, 'add')
    
    def test_merge_invalid_operation(self):
        """Test that invalid merge operation raises ValueError."""
        cube1 = Cube(3)
        cube2 = Cube(3)
        with self.assertRaises(ValueError):
            merge(cube1, cube2, 'invalid')


if __name__ == '__main__':
    unittest.main()
