"""
Basic usage examples for the Cubular API.
"""

from cubular import Cube, rotate, flip, transform
from cubular.utils import create_empty_cube, create_filled_cube, get_slice


def example_basic_cube():
    """Create and manipulate a basic cube."""
    print("=== Basic Cube Example ===")
    
    # Create a 3x3x3 cube
    cube = Cube(3)
    print(f"Created cube: {cube}")
    print(f"Cube size: {cube.size}")
    print(f"Cube data shape: {cube.data.shape}")
    
    # Get and set values
    value = cube.get(0, 0, 0)
    print(f"\nValue at (0,0,0): {value}")
    
    cube.set(0, 0, 0, 999)
    print(f"After setting (0,0,0) to 999: {cube.get(0, 0, 0)}")
    print()


def example_rotations():
    """Demonstrate cube rotations."""
    print("=== Rotation Example ===")
    
    cube = Cube(2)
    print(f"Original cube:\n{cube.data}")
    
    # Rotate around X axis
    cube_x = rotate(cube, 'x', 1)
    print(f"\nAfter rotating around X axis:\n{cube_x.data}")
    
    # Rotate around Y axis
    cube_y = rotate(cube, 'y', 1)
    print(f"\nAfter rotating around Y axis:\n{cube_y.data}")
    
    # Rotate around Z axis
    cube_z = rotate(cube, 'z', 1)
    print(f"\nAfter rotating around Z axis:\n{cube_z.data}")
    print()


def example_flips():
    """Demonstrate cube flips."""
    print("=== Flip Example ===")
    
    cube = Cube(2)
    print(f"Original cube:\n{cube.data}")
    
    # Flip along X axis
    cube_x = flip(cube, 'x')
    print(f"\nAfter flipping along X axis:\n{cube_x.data}")
    
    # Flip along Y axis
    cube_y = flip(cube, 'y')
    print(f"\nAfter flipping along Y axis:\n{cube_y.data}")
    print()


def example_transformations():
    """Demonstrate complex transformations."""
    print("=== Transformation Example ===")
    
    cube = Cube(2)
    print(f"Original cube:\n{cube.data}")
    
    # Apply multiple transformations
    operations = [
        ('rotate', 'x', 1),
        ('flip', 'y'),
        ('rotate', 'z', 2)
    ]
    
    result = transform(cube, operations)
    print(f"\nAfter transformations {operations}:\n{result.data}")
    print()


def example_utility_functions():
    """Demonstrate utility functions."""
    print("=== Utility Functions Example ===")
    
    # Create different types of cubes
    empty = create_empty_cube(3)
    print(f"Empty cube (all zeros):\n{empty.data[0]}")  # Show first slice
    
    filled = create_filled_cube(3, 5)
    print(f"\nFilled cube (all 5s):\n{filled.data[0]}")  # Show first slice
    
    # Get slices
    cube = Cube(3)
    slice_x = get_slice(cube, 'x', 1)
    print(f"\nSlice along X axis at index 1:\n{slice_x}")
    print()


def example_copying():
    """Demonstrate cube copying."""
    print("=== Copy Example ===")
    
    cube1 = Cube(2)
    cube2 = cube1.copy()
    
    print(f"Original cube:\n{cube1.data}")
    print(f"Copied cube:\n{cube2.data}")
    
    # Modify the copy
    cube2.set(0, 0, 0, 999)
    
    print(f"\nAfter modifying copy:")
    print(f"Original cube (unchanged):\n{cube1.data}")
    print(f"Modified copy:\n{cube2.data}")
    print()


if __name__ == "__main__":
    example_basic_cube()
    example_rotations()
    example_flips()
    example_transformations()
    example_utility_functions()
    example_copying()
    
    print("=== All examples completed! ===")
