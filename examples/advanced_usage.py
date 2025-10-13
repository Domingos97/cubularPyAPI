"""
Advanced usage examples for the Cubular API.
Demonstrates more complex operations and patterns.
"""

from cubular import Cube, transform
from cubular.operations import merge
from cubular.utils import create_filled_cube, is_symmetric, get_slice


def example_cube_arithmetic():
    """Demonstrate arithmetic operations between cubes."""
    print("=== Cube Arithmetic Example ===")
    
    # Create two cubes with different values
    cube1 = create_filled_cube(3, 10)
    cube2 = create_filled_cube(3, 5)
    
    print(f"Cube 1 (all 10s):\n{cube1.data[0]}")  # First slice
    print(f"\nCube 2 (all 5s):\n{cube2.data[0]}")  # First slice
    
    # Addition
    sum_cube = merge(cube1, cube2, 'add')
    print(f"\nCube 1 + Cube 2:\n{sum_cube.data[0]}")
    
    # Subtraction
    diff_cube = merge(cube1, cube2, 'subtract')
    print(f"\nCube 1 - Cube 2:\n{diff_cube.data[0]}")
    
    # Multiplication
    mult_cube = merge(cube1, cube2, 'multiply')
    print(f"\nCube 1 * Cube 2:\n{mult_cube.data[0]}")
    print()


def example_complex_transformations():
    """Apply a series of complex transformations."""
    print("=== Complex Transformations Example ===")
    
    cube = Cube(3)
    print(f"Original cube (first slice):\n{cube.data[0]}")
    
    # Create a complex transformation sequence
    transformations = [
        ('rotate', 'x', 1),
        ('rotate', 'y', 1),
        ('flip', 'z'),
        ('rotate', 'z', 2),
    ]
    
    result = transform(cube, transformations)
    print(f"\nAfter complex transformations (first slice):\n{result.data[0]}")
    
    # Show that we can reverse some operations
    reverse_ops = [
        ('rotate', 'z', 2),  # Reverse the z rotation
        ('flip', 'z'),       # Reverse the flip
    ]
    
    partially_reversed = transform(result, reverse_ops)
    print(f"\nPartially reversed (first slice):\n{partially_reversed.data[0]}")
    print()


def example_slicing_analysis():
    """Analyze cube slices."""
    print("=== Slicing Analysis Example ===")
    
    cube = Cube(4)
    
    print("Analyzing slices along X axis:")
    for i in range(cube.size):
        slice_data = get_slice(cube, 'x', i)
        print(f"\nSlice {i}:")
        print(slice_data)
        print(f"Min: {slice_data.min()}, Max: {slice_data.max()}, Mean: {slice_data.mean():.2f}")
    print()


def example_symmetry_detection():
    """Detect symmetry in cubes."""
    print("=== Symmetry Detection Example ===")
    
    # Create an asymmetric cube
    asymmetric = Cube(3)
    print(f"Asymmetric cube is symmetric along X? {is_symmetric(asymmetric, 'x')}")
    print(f"Asymmetric cube is symmetric along Y? {is_symmetric(asymmetric, 'y')}")
    print(f"Asymmetric cube is symmetric along Z? {is_symmetric(asymmetric, 'z')}")
    
    # Create a symmetric cube by merging with itself
    import numpy as np
    # Create a symmetric pattern
    data = np.array([
        [[1, 2, 1], [2, 3, 2], [1, 2, 1]],
        [[2, 3, 2], [3, 4, 3], [2, 3, 2]],
        [[1, 2, 1], [2, 3, 2], [1, 2, 1]]
    ])
    symmetric = Cube(3, data)
    
    print(f"\nSymmetric cube is symmetric along X? {is_symmetric(symmetric, 'x')}")
    print(f"Symmetric cube is symmetric along Y? {is_symmetric(symmetric, 'y')}")
    print(f"Symmetric cube is symmetric along Z? {is_symmetric(symmetric, 'z')}")
    print()


def example_cube_pipeline():
    """Demonstrate a processing pipeline."""
    print("=== Cube Processing Pipeline Example ===")
    
    # Start with a base cube
    cube = Cube(3)
    print("Step 1: Create initial cube")
    print(f"Sum of all elements: {cube.data.sum()}")
    
    # Step 2: Rotate
    step1 = transform(cube, [('rotate', 'x', 1)])
    print(f"\nStep 2: After rotation")
    print(f"Sum of all elements: {step1.data.sum()}")
    
    # Step 3: Merge with another cube
    other = create_filled_cube(3, 1)
    step2 = merge(step1, other, 'add')
    print(f"\nStep 3: After merging with ones")
    print(f"Sum of all elements: {step2.data.sum()}")
    
    # Step 4: Apply more transformations
    step3 = transform(step2, [('flip', 'y'), ('rotate', 'z', 1)])
    print(f"\nStep 4: After flip and rotate")
    print(f"Sum of all elements: {step3.data.sum()}")
    
    print(f"\nFinal cube (first slice):\n{step3.data[0]}")
    print()


def example_performance():
    """Test with larger cubes."""
    print("=== Performance with Larger Cubes Example ===")
    
    import time
    
    # Test with different sizes
    for size in [5, 10, 20]:
        start = time.time()
        cube = Cube(size)
        create_time = time.time() - start
        
        start = time.time()
        rotated = transform(cube, [
            ('rotate', 'x', 1),
            ('rotate', 'y', 1),
            ('rotate', 'z', 1)
        ])
        transform_time = time.time() - start
        
        print(f"Cube size {size}x{size}x{size}:")
        print(f"  Creation time: {create_time*1000:.2f}ms")
        print(f"  Transform time: {transform_time*1000:.2f}ms")
        print(f"  Total elements: {size**3}")
    print()


if __name__ == "__main__":
    example_cube_arithmetic()
    example_complex_transformations()
    example_slicing_analysis()
    example_symmetry_detection()
    example_cube_pipeline()
    example_performance()
    
    print("=== All advanced examples completed! ===")
