#!/usr/bin/env python3
"""
Python file header checker for Tactics2D project.
This hook checks Python files for compliance with the header style guide.
"""

import json
import os
import sys


def check_python_file(file_path):
    """Check a Python file for header compliance."""
    if not os.path.exists(file_path):
        return

    try:
        with open(file_path, encoding="utf-8") as f:
            lines = [f.readline() for _ in range(10)]
    except Exception:
        return

    warnings = []

    # Check for SPDX license identifier
    spdx_found = any("SPDX-License-Identifier: GPL-3.0-or-later" in line for line in lines)
    if not spdx_found:
        warnings.append(
            "Missing SPDX license identifier. Add: # SPDX-License-Identifier: GPL-3.0-or-later"
        )

    # Check for copyright notice
    copyright_found = any("Copyright (C)" in line for line in lines)
    if not copyright_found:
        current_year = 2026  # Could use datetime.date.today().year
        warnings.append(
            f"Missing copyright notice. Add: # Copyright (C) {current_year}, Tactics2D Authors. Released under the GNU GPLv3."
        )

    # Check for module docstring (triple quotes)
    docstring_found = any(line.strip().startswith('"""') for line in lines)
    if not docstring_found:
        warnings.append(
            "Missing module docstring. Add a brief module description in triple quotes."
        )

    if warnings:
        print(f"Warning: Python file {file_path} header issues:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
        print("Please refer to .claude/.python_header_style.md for style guide.", file=sys.stderr)


def main():
    # Read hook input from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            return

        data = json.loads(input_data)
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        # For Write and Edit tools, check if the file is a Python file
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            if file_path and file_path.endswith(".py"):
                check_python_file(file_path)
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error in hook: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
