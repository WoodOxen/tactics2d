#!/usr/bin/env python3
"""
Pytest style checker for Tactics2D project.
This hook checks test files for compliance with pytest style guidelines.
"""

import ast
import json
import os
import sys
from pathlib import Path


def is_test_file(file_path):
    """Check if file is a pytest test file."""
    path = Path(file_path)
    return path.name.startswith("test_") and path.name.endswith(".py")


def analyze_test_file(file_path):
    """Analyze a test file for style compliance."""
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []

    warnings = []

    # Parse AST to analyze structure
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return warnings

    # Check for test functions
    test_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("test_"):
                test_functions.append(node.name)
                # Check function docstring
                if not ast.get_docstring(node):
                    warnings.append(
                        f"Test function '{node.name}' missing docstring. Add: '''Test {node.name[5:].replace('_', ' ')}.'''"
                    )

    # Check imports (basic check)
    lines = content.split("\n")
    import_sections = {"sys_path": False, "stdlib": False, "third_party": False, "project": False}

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Check for sys.path modifications
        if "sys.path.append" in line:
            import_sections["sys_path"] = True

        # Check import order (simplistic check)
        if line_stripped.startswith("import ") or line_stripped.startswith("from "):
            if "unittest" in line or "sys" in line or "os" in line or "logging" in line:
                import_sections["stdlib"] = True
            elif "numpy" in line or "pytest" in line or "shapely" in line or "matplotlib" in line:
                import_sections["third_party"] = True
                if not import_sections["stdlib"]:
                    warnings.append(
                        "Third-party imports appear before standard library imports. Standard libs should come first."
                    )
            elif "tactics2d" in line:
                import_sections["project"] = True
                if not import_sections["third_party"]:
                    warnings.append(
                        "Project imports appear before third-party imports. Third-party libs should come before project modules."
                    )

    # Check for pytest markers
    has_pytest_markers = any("@pytest.mark." in line for line in lines)
    if not has_pytest_markers and test_functions:
        warnings.append(
            "All test functions must have pytest markers (e.g., @pytest.mark.env). "
            "Add appropriate markers for test categorization."
        )

    return warnings


def main():
    # Read hook input from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            return

        data = json.loads(input_data)
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        # For Write and Edit tools, check if the file is a test file
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            if file_path and is_test_file(file_path):
                warnings = analyze_test_file(file_path)
                if warnings:
                    print(f"Pytest style suggestions for {file_path}:", file=sys.stderr)
                    for warning in warnings:
                        print(f"  - {warning}", file=sys.stderr)
                    print(
                        "Please refer to .claude/pytest_style_guide.md for style guidelines.",
                        file=sys.stderr,
                    )
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error in pytest style hook: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
