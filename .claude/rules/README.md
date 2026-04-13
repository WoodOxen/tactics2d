# Claude Rules for Tactics2D

This directory contains rule definitions for Claude Code when working with the Tactics2D project.

## Current Rules

### `python_header.json`
Enforces Python file header style according to the project standard.

**Features:**
- Standard copyright and license header
- File-type-specific description patterns
- Proper noun preservation
- Validation and suggestion actions

**Format:**
```python
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Module description."""
```

**Description Patterns:**
- `__init__.py`: `"{module_name} module."`
- `test_*.py`: `"Tests for {test_subject}."`
- `parse_*.py`: `"{dataset_name} parser implementation."`
- `envs/*.py`: `"{environment_type} environment implementation."`
- `*2*.py`: `"{source} to {target} converter implementation."`
- `generate_*.py`: `"{what} generator implementation."`
- `*controller.py`: `"{controller_type} controller implementation."`
- Default: `"{function_name} implementation."`

## Configuration

Rules are enabled in `.claude/settings.json`:

```json
"rules": {
  "enabled": true,
  "directories": [".claude/rules"]
}
```

## Integration

- **Hook**: A `PostToolUse` hook calls `check_python_header.py` to validate Python files after Write/Edit operations.
- **Style Guide**: Detailed specifications in `.claude/.python_header_style.md`.

## Adding New Rules

1. Create a new JSON rule file in this directory
2. Define rule properties (name, description, filePatterns, actions)
3. Ensure JSON syntax is valid
4. Rules will be loaded automatically when Claude starts

## Testing

To test rule validation manually:

```bash
python3 .claude/hooks/check_python_header.py < test_input.json
```

Example test input:

```json
{
  "tool_name": "Write",
  "tool_input": {
    "file_path": "test.py"
  }
}
```
