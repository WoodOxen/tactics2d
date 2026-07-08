# Tactics2D Code Style Guide

## Copyright Header (all files)

Every Python file starts with the standard header; only the year varies:

```python
# Copyright (C) <YEAR>, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""<description>."""
```

Use the **first year** the file was created (not the year of latest edit).

### Year Reference

| Year | Files |
|------|-------|
| 2023 | sensor_base, camera, participant_base, vehicle, map/*, physics/* |
| 2024 | trajectory, state, parse_womd, scenario_manager |
| 2025 | renderers/matplotlib/renderer |
| 2026 | behavior/*, renderers/config, routing/*, geometry/* |

### Description Patterns

- `__init__.py`: `"{module_name} module."`
- `test_*.py`: `"Tests for {test_subject}."`
- `parse_*.py`: `"{dataset_name} parser implementation."`
- `envs/*.py`: `"{environment_type} environment implementation."`
- `*2*.py`: `"{source} to {target} converter implementation."`
- `generate_*.py`: `"{what} generator implementation."`
- `*controller.py`: `"{controller_type} controller implementation."`
- Default: `"{function_name} implementation."`

## Docstring Checklist

### Notebooks (`.ipynb`) — no docstrings in code cells

Notebooks are tutorial/demonstration documents, not library code. All explanation
belongs in **markdown cells**. Code cells should contain only runnable code —
no `Args:`, `Returns:`, `"""` docstrings, or other API documentation that
belongs in `.py` files. The one exception is a single-line `#` comment for
clarity when the code itself is non-obvious.

### 1. Use Google-style, not NumPy/reST

| Style | Hit count | Verdict |
|-------|-----------|---------|
| `Args:` | 397 | ✅ **Standard — input** |
| `Returns:` | 282 | ✅ **Standard — output** |
| `Raises:` | 68 | ✅ **Standard — exceptions** |
| `Parameters` / `----------` | 0 (fixed) | ❌ NumPy style — use `Args:` instead |
| `:param` / `:return` / `:raises` | 0 | ❌ reST style — use `Args:` / `Returns:` / `Raises:` instead |

### 2. Class docstrings — `Attributes:` block

```python
class Vehicle(ParticipantBase):
    """This class defines a four-wheeled vehicle with its common properties.

    Attributes:
        id_ (int): The unique identifier of the vehicle.
        type_ (str): The type of the vehicle. Defaults to "medium_car".
        trajectory (Trajectory): The trajectory of the vehicle.
        color (tuple): The color of the vehicle. This attribute is **read-only**.
    """
```

- Mark computed/read-only attributes with `**read-only**`.
- Omit `Attributes:` on trivial `@dataclass` classes (brief prose is fine).

### 3. Method docstrings — `Args:` / `Returns:` / `Raises:`

```python
def update(self, frame, participants, ...):
    """Short sentence describing what this method does.

    Longer explanation if needed (blank line after summary).

    Args:
        frame (int): The frame number. The unit is millisecond (ms).
        participants (Dict): All participants in the scenario.
        interval (int, optional): Time interval. Defaults to None.

    Returns:
        A new state of the traffic participant.

    Raises:
        ValueError: If the input cannot be converted.
    """
```

**General rules:**
- **Summary line**: Imperative mood ("Return the ...", "Update the ..."). Avoid "This function..." for new code (legacy may have it).
- **Defaults**: Include "Defaults to ..." at end of each parameter description.
- **Units**: Mention units for physical quantities (m, m/s, rad, ms).
- **Raises**: Only when the method explicitly raises.

**`Args:` rules:**
- Format: `name (type): Description.` — aligned colons.
- Use `, optional` in the type for parameters with defaults (e.g. `(str, optional)`).
- One line per argument.

**`Returns:` rules:**
- Omit the type if the function signature already has a return type annotation (just describe what's returned).
- Include the type only when there's no annotation (rare in this codebase).
- Describe the return value's meaning, not its mechanics.
- For methods that return `None`, omit `Returns:` entirely (unless the None is meaningful, e.g. `Returns: None. Updates the internal state in place.`).
- Use plural if returning a collection: `Returns: A list of matched participants.`
- If returning a tuple of multiple values, enumerate them:
  ```python
  Returns:
      A tuple of:
          - x (float): The x-coordinate in meters.
          - y (float): The y-coordinate in meters.
  ```

**`Raises:` rules:**
- Format: `ValueError: Description of when this exception is raised.`
- Only document exceptions that are explicitly raised by the method, not those propagated from callees.

### 4. Property docstrings — brief one-liner

```python
@property
def last_state(self):
    """Return the last state of the trajectory. None if empty."""
```

### 5. No `:param:` or `Parameters\n----------`

Neither reStructuredText `:param:` nor NumPy `Parameters\n----------` are used anywhere in this codebase. Always use Google-style `Args:`.

### 6. Admonitions (optional, for extra clarity)

```python
!!! info "TODO"
    More physics models will be added in the future.

!!! quote "Reference"
    Official Waymo Open Dataset: https://waymo.com/open/
```

### 7. LaTeX — use `r"""..."""` raw strings

```python
r"""Initialize the vehicle.

Args:
    max_steer (float): The maximum approach angle. Unit is radian. Defaults to $\pi$/6.
"""
```

### 8. Module-level constants / configs — inline `#` comments

```python
# default color for lane class subtypes
"lane": "black",
```

### 9. `# ---` section separators

Allowed in renderers, converters, and long implementation files:

```python
# ------------------------------------------------------------------
# Trajectory gradient overlay (zorder=5, above road, below vehicles)
# ------------------------------------------------------------------
```

## Inconsistent Patterns (what the 2026 files do differently)

Some newer files (`behavior/limsim/`, `geometry/`, `routing/`) use brief plain-prose docstrings without `Args:`/`Returns:` sections. **Legacy code is accepted as-is**, but **new code and edits to existing files should match the surrounding file's style**:

- Edits to a 2023-2024 file → full Google-style with `Args:`/`Returns:`/`Raises:`
- Edits to a 2026 file → follow the local style (brief or full depending on what's already there)
- **New files** → prefer full Google-style for clarity

## Quick Reference (New Method Template)

```python
def method_name(self, param1, param2=None):
    """Short description in imperative mood.

    Args:
        param1 (int): Description. The unit is meter.
        param2 (str, optional): Description. Defaults to None.

    Returns:
        Description of the return value.

    Raises:
        ValueError: If param1 is negative.
    """
```

---

## CHANGELOG Format

CHANGELOG.md follows the [Keep a Changelog](https://keepachangelog.com/) standard:

```markdown
# Change Log

## [Unreleased]

### Added
- New feature description

### Changed
- Change description

### Fixed
- Bug fix description

### Removed
- Removal description
```

**Rules:**
- Must have `## [Unreleased]` section
- Version headers: `## [version] - YYYY-MM-DD`
- Section headers: `### Added`, `### Changed`, `### Fixed`, `### Removed` only
- Entries start with `- ` (hyphen space)
- Add changes to `[Unreleased]` **before** coding, not after
- Move to version section after release

A PostToolUse hook automatically validates format and extracts PR descriptions from the `[Unreleased]` section when running `gh pr create`.

---

## Test Style

### File Header

```python
# Copyright (C) <YEAR>, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for {module_or_class}."""
```

### Import Organization

Grouped in order: standard library → third-party → `tactics2d`:

```python
import os
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from shapely.geometry import Point

from tactics2d.map.element import Map
```

### Test Structure

**Preferred** — function-style tests with pytest markers:

```python
@pytest.mark.sensor
def test_lidar_initialization_default_params(mock_map):
    """Test lidar initialization with default parameters."""
    lidar = SingleLineLidar(id_=1, map_=mock_map)
    assert lidar.id_ == 1
    assert lidar.perception_range == 12.0
```

**Acceptable** — class structure for legacy tests or organizing related fixtures:

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

### Fixtures

- Module-level for test-specific fixtures; `conftest.py` for shared ones across the project.
- Factory pattern for parameterized fixtures:

```python
@pytest.fixture
def create_lidar():
    """Factory fixture for creating lidar instances."""
    def _create_lidar(id_, perception_range=12.0):
        mock_map = Mock(spec=Map)
        mock_map.boundary = (0, 0, 100, 100)
        return SingleLineLidar(id_=id_, map_=mock_map, perception_range=perception_range)
    return _create_lidar
```

### Naming Conventions

| Element | Pattern | Example |
|---------|---------|---------|
| File | `test_{module}.py` | `test_lidar.py` |
| Class | `Test{ClassName}` | `TestSingleLineLidar` |
| Function | `test_{behavior}_{condition}` | `test_initialization_with_default_params` |
| Fixture | `{resource_name}` | `mock_map` |

### Docstrings

| Element | Format |
|---------|--------|
| Class | `"""Test suite for {ClassName} class."""` |
| Method/Function | `"""Test {behavior} under {condition}."""` |
| Fixture | `"""Create {resource} for testing."""` |

### Assertions

```python
# Simple equality
assert result == expected

# Type checking
assert isinstance(obj, dict)

# Collection checks
assert len(items) == 5
assert "key" in dictionary

# Numerical approximations
np.testing.assert_array_almost_equal(actual, expected, decimal=6)

# Exception testing
with pytest.raises(ValueError, match="expected error message"):
    function_that_raises()
```

### Pytest Markers

**All test functions must have at least one pytest marker.** Common markers from `tests/pytest.ini`:

| Marker | Purpose |
|--------|---------|
| `env` | Environment tests |
| `math` | Mathematics calculation tests |
| `dataset_parser` | Dataset parsing tests |
| `map_element` | Map element tests |
| `map_generator` | Map generation tests |
| `map_parser` | Map parsing tests |
| `participant` | Participant tests |
| `physics` | Physics simulation tests |
| `render` | Render-related tests |
| `search` | Search algorithms and utilities tests |

Multiple markers can be combined for tests that span categories.

### Mocking

```python
from unittest.mock import Mock, MagicMock, patch

# Mock with spec for type safety
mock_map = Mock(spec=Map)
mock_map.boundary = (0, 0, 100, 100)

# Configure method returns
mock_method = Mock(return_value=42)

# Patch as context manager
with patch("module.ClassName.method") as mock_method:
    mock_method.return_value = expected
    # test code
```

### Temporary Files

Use pytest's `tmp_path` fixture:

```python
def test_file_creation(tmp_path):
    """Test file creation in temporary directory."""
    output_file = tmp_path / "output.txt"
    output_file.write_text("test content")
    assert output_file.exists()
```

### Code Formatting

- Line length: 100 characters max (configured in `black`)
- Indent: 4 spaces per level
- Multiline args: vertical alignment

```python
lidar = SingleLineLidar(
    id_=1,
    map_=mock_map,
    perception_range=20.0,
)
```

### Best Practices

1. **One assertion per test behavior** — focus each test on one specific behavior.
2. **Descriptive names** — `test_{behavior}_{condition}`.
3. **Fixtures for setup** — not `setUp`/`tearDown` methods.
4. **Independent** — tests must not depend on each other.
5. **Deterministic** — same result every run.
6. **Fast** — mock slow dependencies.
7. **Markers required** — every test function needs `@pytest.mark.<marker>`.
8. **Meaningful coverage** — not just line count.

### Verification

```bash
black --line-length 100 tests/
isort --profile black tests/
```

---

## Rule System (`.claude/rules/`)

The project uses structured JSON rules under `.claude/rules/` for automated code checks:

| File | Purpose |
|------|---------|
| `python_header.json` | Python file header format validation |
| `pytest_style.json` | Pytest test style guidelines |
| `changelog_check.json` | CHANGELOG format & PR description extraction |

These rules feed PostToolUse hooks that run after file edits. To add a new rule:
1. Create a JSON file in `.claude/rules/`
2. Define `name`, `description`, `filePatterns`, and `actions`
3. Enable in `.claude/settings.json` via `"rules"."directories"`
