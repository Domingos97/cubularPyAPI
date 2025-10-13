# cubularPyAPI

A Python API for manipulating and performing operations on 3D cubes. This library provides a simple and intuitive interface for creating, transforming, and analyzing cubic data structures.

## Features

- **Cube Creation**: Create cubes with custom sizes and initial data
- **Transformations**: Rotate and flip cubes along any axis
- **Operations**: Merge cubes with various operations (add, subtract, multiply, max, min)
- **Utilities**: Helper functions for validation, slicing, and symmetry checking
- **Type Safety**: Full type hints for better IDE support
- **Well Tested**: Comprehensive test suite included

## Installation

### From source
```bash
git clone https://github.com/Domingos97/cubularPyAPI.git
cd cubularPyAPI
pip install -r requirements.txt
pip install -e .
```

### Requirements
- Python 3.7+
- NumPy >= 1.19.0

## Quick Start

```python
from cubular import Cube, rotate, flip, transform

# Create a 3x3x3 cube
cube = Cube(3)

# Get and set values
value = cube.get(0, 0, 0)
cube.set(0, 0, 0, 999)

# Rotate the cube around the X axis
rotated = rotate(cube, 'x', 1)

# Flip the cube along the Y axis
flipped = flip(cube, 'y')

# Apply multiple transformations
operations = [('rotate', 'x', 1), ('flip', 'y')]
transformed = transform(cube, operations)
```

## API Reference

### Cube Class

The main class for representing 3D cubes.

```python
from cubular import Cube

# Create a cube with default sequential values
cube = Cube(size=3)

# Create a cube with custom data
import numpy as np
data = np.ones((3, 3, 3))
cube = Cube(size=3, data=data)
```

#### Methods

- `get(x, y, z)`: Get value at coordinates
- `set(x, y, z, value)`: Set value at coordinates
- `rotate_x(k)`: Rotate around X axis k times (90° each)
- `rotate_y(k)`: Rotate around Y axis k times
- `rotate_z(k)`: Rotate around Z axis k times
- `flip_x()`: Flip along X axis
- `flip_y()`: Flip along Y axis
- `flip_z()`: Flip along Z axis
- `copy()`: Create a deep copy of the cube

### Operations

Functions for manipulating cubes.

```python
from cubular.operations import rotate, flip, transform, merge

# Rotate a cube
rotated = rotate(cube, axis='x', k=1)

# Flip a cube
flipped = flip(cube, axis='y')

# Apply multiple transformations
ops = [('rotate', 'x', 1), ('flip', 'y')]
result = transform(cube, ops)

# Merge two cubes
cube1 = Cube(3)
cube2 = Cube(3)
merged = merge(cube1, cube2, operation='add')
```

### Utilities

Helper functions for common tasks.

```python
from cubular.utils import (
    create_empty_cube,
    create_filled_cube,
    get_slice,
    get_dimensions,
    is_symmetric
)

# Create special cubes
empty = create_empty_cube(3)  # All zeros
filled = create_filled_cube(3, value=5)  # All 5s

# Get a 2D slice
slice_data = get_slice(cube, axis='x', index=1)

# Check dimensions
dims = get_dimensions(cube)  # Returns (3, 3, 3)

# Check symmetry
symmetric = is_symmetric(cube, axis='x')  # Returns bool
```

## Examples

Check out the `examples/` directory for more detailed usage examples:

```bash
python examples/basic_usage.py
```

## Testing

Run the test suite:

```bash
python -m unittest discover tests
```

Or test individual modules:

```bash
python -m unittest tests.test_cube
python -m unittest tests.test_operations
python -m unittest tests.test_utils
```

## Development

### Project Structure

```
cubularPyAPI/
├── cubular/              # Main package
│   ├── __init__.py      # Package initialization
│   ├── cube.py          # Cube class
│   ├── operations.py    # Transformation operations
│   └── utils.py         # Utility functions
├── tests/               # Test suite
│   ├── test_cube.py
│   ├── test_operations.py
│   └── test_utils.py
├── examples/            # Usage examples
│   └── basic_usage.py
├── setup.py             # Package setup
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Author

Domingos97