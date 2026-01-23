# Pytest Test Code Style Guide for Tactics2D

This document outlines the code style guidelines for pytest test files in the Tactics2D project. Consistent test code style improves readability, maintainability, and collaboration.

## File Structure

### 1. File Header
All test files must include the standard copyright header:

```python
# Copyright (C) {year}, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for {test_subject}."""
```

For test files, the docstring should follow the pattern: `"""Tests for {module_or_class}."""`

### 2. Import Organization
Imports should be grouped and ordered as follows:

```python
# 1. System path modifications (if needed)
import sys
sys.path.append(".")
sys.path.append("..")

# 2. Standard library imports
import os
import time
import logging
from unittest.mock import Mock, patch

# 3. Third-party library imports
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

# 4. Project internal imports
from tactics2d.map.element import Map
from tactics2d.sensor.lidar import SingleLineLidar
```

### 3. Test Structure Preference
**Preferred**: Use function-style tests with pytest markers for better modularity:

```python
@pytest.mark.sensor
def test_lidar_initialization_default_params(mock_map):
    """Test lidar initialization with default parameters."""
    lidar = SingleLineLidar(id_=1, map_=mock_map)
    assert lidar.id_ == 1
    assert lidar.map_ == mock_map
    assert lidar.perception_range == 12.0

@pytest.mark.sensor
@pytest.mark.parametrize("perception_range", [10.0, 20.0, 30.0])
def test_lidar_initialization_custom_params(mock_map, perception_range):
    """Test lidar initialization with custom parameters."""
    lidar = SingleLineLidar(id_=2, map_=mock_map, perception_range=perception_range)
    assert lidar.id_ == 2
    assert lidar.perception_range == perception_range
```

**Acceptable**: Class structure for legacy tests or when organizing related fixtures:

```python
class TestSingleLineLidar:
    """Test suite for SingleLineLidar class."""

    @pytest.fixture
    def mock_map(self):
        """Create a mock map for testing."""
        mock_map = Mock(spec=Map)
        mock_map.boundary = (0, 0, 100, 100)
        return mock_map

    @pytest.mark.sensor
    def test_initialization(self, mock_map):
        """Test lidar initialization with different parameters."""
        lidar = SingleLineLidar(id_=1, map_=mock_map)
        assert lidar.id_ == 1
```

### 4. Fixture Organization

In function-style tests, fixtures can be organized at different levels:

#### Module-level Fixtures
```python
import pytest

@pytest.fixture
def mock_map():
    """Create a mock map for testing."""
    mock_map = Mock(spec=Map)
    mock_map.boundary = (0, 0, 100, 100)
    return mock_map

@pytest.fixture
def temporary_directory(tmp_path):
    """Create a temporary directory for file operations."""
    return tmp_path / "test_output"
```

#### Shared Fixtures (conftest.py)
Place common fixtures in `tests/conftest.py` for project-wide reuse:
```python
# tests/conftest.py
import pytest

@pytest.fixture
def default_vehicle_params():
    """Default parameters for vehicle testing."""
    return {
        "id_": 1,
        "type_": "car",
        "length": 4.0,
        "width": 1.8,
        "height": 1.5,
        "physics_model": "kinematic_bicycle",
    }
```

#### Fixture Factory Pattern
For fixtures that require parameters or complex setup:
```python
@pytest.fixture
def create_lidar():
    """Factory fixture for creating lidar instances."""
    def _create_lidar(id_, perception_range=12.0):
        mock_map = Mock(spec=Map)
        mock_map.boundary = (0, 0, 100, 100)
        return SingleLineLidar(id_=id_, map_=mock_map, perception_range=perception_range)
    return _create_lidar

@pytest.mark.sensor
def test_lidar_factory(create_lidar):
    """Test using a fixture factory."""
    lidar = create_lidar(id_=3, perception_range=15.0)
    assert lidar.id_ == 3
    assert lidar.perception_range == 15.0
```

## Naming Conventions

### Test Files
- Pattern: `test_{module_name}.py` or `test_{class_name}.py`
- Examples: `test_lidar.py`, `test_dataset_parser.py`

### Test Classes
- Pattern: `Test{ClassName}`
- Examples: `TestSingleLineLidar`, `TestMatplotlibRenderer`

### Test Methods
- Pattern: `test_{behavior}_{condition}`
- Examples:
  - `test_initialization_with_default_params`
  - `test_update_method_signature`
  - `test_error_handling_invalid_input`

### Fixture Methods
- Pattern: `{resource_name}`
- Examples: `mock_map`, `realistic_participants`, `temp_directory`

## Documentation

### Class Docstrings
- Format: `"""Test suite for {ClassName} class."""`
- Example: `"""Test suite for SingleLineLidar class."""`

### Method Docstrings
- Format: `"""Test {behavior} under {condition}."""`
- Examples:
  - `"""Test initialization with default parameters."""`
  - `"""Test update method with None parameters."""`

### Fixture Docstrings
- Format: `"""Create {resource} for testing."""`
- Example: `"""Create mock map for testing."""`

## Assertion Style

### Preferred Assertions
```python
# Simple equality
assert result == expected

# Type checking
assert isinstance(obj, dict)

# Collection checks
assert len(items) == 5
assert "key" in dictionary

# Boolean conditions
assert is_valid
assert not has_error
```

### Numerical Approximations
```python
import numpy as np
np.testing.assert_array_almost_equal(actual, expected, decimal=6)
```

### Exception Testing
```python
import pytest

with pytest.raises(ValueError, match="expected error message"):
    function_that_raises()
```

## Code Formatting

### Line Length
- Maximum 100 characters per line (configured in black)

### Indentation
- 4 spaces per indentation level

### Multiline Arguments
```python
# Vertical alignment for long argument lists
lidar = SingleLineLidar(
    id_=1,
    map_=mock_map,
    perception_range=20.0,
    freq_scan=5.0,
    freq_detect=10000.0,
)
```

### Decorators
```python
@pytest.mark.parametrize(
    "dataset, expected",
    [
        ("highD", 1047),
        ("inD", 384),
    ],
)
def test_dataset_parser(dataset, expected):
    # test code
```

## Pytest Markers

**All test functions must have at least one pytest marker.** Use appropriate markers from `tests/pytest.ini` for test categorization:

```python
@pytest.mark.env  # Environment tests
@pytest.mark.math  # Mathematics calculation tests
@pytest.mark.dataset_parser  # Dataset parsing tests
@pytest.mark.map_element  # Map element tests
@pytest.mark.map_generator  # Map generation tests
@pytest.mark.map_parser  # Map parsing tests
@pytest.mark.participant  # Participant tests
@pytest.mark.physics  # Physics simulation tests
@pytest.mark.render  # Render-related tests
@pytest.mark.search  # Search algorithms and utilities tests
```

**Important:**
- Markers help organize tests and enable selective test execution (e.g., `pytest -m env`)
- Each test function should have at least one marker matching its category
- Multiple markers can be combined for tests that span categories
- Test functions without markers will generate style warnings

## Mock Objects

Use `unittest.mock` for test doubles:

```python
from unittest.mock import Mock, MagicMock, patch

# Create mock with spec for type safety
mock_map = Mock(spec=Map)
mock_map.boundary = (0, 0, 100, 100)

# Configure method returns
mock_method = Mock(return_value=42)

# Use patch for context managers
with patch('module.ClassName.method') as mock_method:
    mock_method.return_value = expected
    # test code
```

## Temporary Files and Directories

Use pytest's `tmp_path` fixture:

```python
def test_file_creation(tmp_path):
    """Test file creation in temporary directory."""
    output_file = tmp_path / "output.txt"
    output_file.write_text("test content")
    assert output_file.exists()
```

## Logging in Tests

Configure logging appropriately:

```python
import logging
logging.basicConfig(level=logging.INFO)

def test_with_logging():
    """Test with logging output."""
    logging.info("Starting test")
    # test code
    logging.info("Test completed")
```

## Best Practices

1. **One Assertion per Test**: Focus each test on one specific behavior
2. **Descriptive Names**: Use clear, descriptive test names
3. **Setup/Teardown**: Use fixtures for reusable test setup
4. **Independence**: Tests should not depend on each other
5. **Determinism**: Tests should produce the same results every time
6. **Speed**: Keep tests fast; use mocks for slow dependencies
7. **Coverage**: Aim for meaningful test coverage, not just line count

## Example Test File

```python
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for SingleLineLidar."""

import sys
sys.path.append(".")
sys.path.append("..")

from unittest.mock import Mock
import numpy as np
import pytest
from shapely.geometry import Point

from tactics2d.map.element import Map
from tactics2d.sensor.lidar import SingleLineLidar


# Module-level fixtures
@pytest.fixture
def mock_map():
    """Create a mock map for testing."""
    mock_map = Mock(spec=Map)
    mock_map.boundary = (0, 0, 100, 100)
    mock_map.areas = {}
    mock_map.lanes = {}
    return mock_map

@pytest.fixture
def create_lidar(mock_map):
    """Factory fixture for creating lidar instances."""
    def _create_lidar(id_, perception_range=12.0, freq_scan=10.0, freq_detect=10000.0):
        return SingleLineLidar(
            id_=id_,
            map_=mock_map,
            perception_range=perception_range,
            freq_scan=freq_scan,
            freq_detect=freq_detect,
        )
    return _create_lidar


# Test functions with pytest markers
@pytest.mark.sensor
def test_lidar_initialization_default_params(mock_map):
    """Test lidar initialization with default parameters."""
    lidar = SingleLineLidar(id_=1, map_=mock_map)
    assert lidar.id_ == 1
    assert lidar.map_ == mock_map
    assert lidar.perception_range == 12.0

@pytest.mark.sensor
@pytest.mark.parametrize("perception_range, freq_scan", [(10.0, 5.0), (20.0, 10.0), (30.0, 15.0)])
def test_lidar_initialization_custom_params(mock_map, perception_range, freq_scan):
    """Test lidar initialization with custom parameters."""
    lidar = SingleLineLidar(
        id_=2,
        map_=mock_map,
        perception_range=perception_range,
        freq_scan=freq_scan,
    )
    assert lidar.id_ == 2
    assert lidar.perception_range == perception_range
    assert lidar.freq_scan == freq_scan

@pytest.mark.sensor
def test_lidar_factory_pattern(create_lidar):
    """Test lidar creation using factory fixture."""
    lidar = create_lidar(id_=3, perception_range=15.0)
    assert lidar.id_ == 3
    assert lidar.perception_range == 15.0

@pytest.mark.sensor
@pytest.mark.render
def test_lidar_visualization(mock_map, tmp_path):
    """Test lidar visualization output."""
    lidar = SingleLineLidar(id_=4, map_=mock_map)
    # Visualization test implementation would go here
    # Example: Save visualization to temporary directory
    output_file = tmp_path / "lidar_visualization.png"
    # In actual test: lidar.visualize(output_file)
    # assert output_file.exists()
    pass
```

## Verification

To check if your test file follows these guidelines:

1. Run black formatting: `black --line-length 100 tests/test_your_file.py`
2. Run isort: `isort --profile black tests/test_your_file.py`
3. Review import order and structure
4. Verify test names and docstrings
5. Ensure appropriate pytest markers are used

## References

- [pytest documentation](https://docs.pytest.org/)
- [Python unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Tactics2D Python Header Style Guide](../.python_header_style.md)
- [Black Code Style](https://black.readthedocs.io/)
