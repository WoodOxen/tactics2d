#!/usr/bin/env python3
"""
Pytest style checker for Tactics2D project.

Reads formatting rules from .claude/rules/pytest_style.json and
validates test files after Write/Edit operations.
"""

import ast
import json
import os
import sys
from pathlib import Path

from rule_loader import load_rule

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _is_test_file(file_path: str) -> bool:
    path = Path(file_path)
    return path.name.startswith("test_") and path.name.endswith(".py")


def _get_module_name(lines: list, i: int) -> str:
    """Return a human-readable name for an import node for reporting."""
    return lines[i].strip()


# --------------------------------------------------------------------------
# Validation logic
# --------------------------------------------------------------------------


def analyze_test_file(file_path: str, rule: dict) -> list:
    """Analyze a test file for style compliance using *rule*.

    Returns a list of warning strings (empty if compliant).
    """
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return []

    warnings = []
    lines = content.split("\n")

    # --- Parse AST ---
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return warnings

    # --- Config from rule ---
    import_cfg = rule.get("importOrder", {})
    stdlib_modules = set(import_cfg.get("standardLibs", []))
    third_party_modules = set(import_cfg.get("thirdPartyLibs", []))
    project_modules = set(import_cfg.get("projectLibs", []))
    doc_cfg = rule.get("documentation", {})
    marker_cfg = rule.get("pytestMarkers", {})
    markers_required = marker_cfg.get("required", True)
    common_markers = set(marker_cfg.get("commonMarkers", []))
    struct_cfg = rule.get("testStructure", {})

    # --- Collect test functions ---
    test_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_functions.append(node)

    # --- Check test function docstrings ---
    method_doc_template = doc_cfg.get("methodDocstring", "")
    for fn in test_functions:
        if not ast.get_docstring(fn):
            if method_doc_template:
                suggested = method_doc_template.replace(
                    "{behavior}", fn.name[5:].replace("_", " ")
                ).replace("{condition}", "")
                warnings.append(
                    f"Test function '{fn.name}' missing docstring. "
                    f'Expected something like: """{suggested}"""'
                )
            else:
                warnings.append(f"Test function '{fn.name}' missing docstring.")

    # --- Check import order ---
    seen_stdlib = False
    seen_third_party = False
    seen_project = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("import ") or stripped.startswith("from "):
            # Determine category of this import.
            is_stdlib = any(mod in stripped for mod in stdlib_modules)
            is_third_party = any(mod in stripped for mod in third_party_modules)
            is_project = any(mod in stripped for mod in project_modules)

            if is_stdlib and not seen_stdlib:
                seen_stdlib = True
            elif is_third_party:
                if not seen_stdlib and not is_project:
                    # Third-party before stdlib.
                    pass  # The next block handles ordering.
                if not seen_stdlib and not seen_third_party:
                    # First third-party import but stdlib not seen yet.
                    warnings.append(
                        f"Third-party import '{_get_module_name(lines, i)}' "
                        f"appears before standard library imports. "
                        f"Order: standard libs -> third-party -> tactics2d."
                    )
                seen_third_party = True
            elif is_project:
                if not seen_third_party and not is_stdlib:
                    warnings.append(
                        f"Project import '{_get_module_name(lines, i)}' "
                        f"appears before third-party imports. "
                        f"Order: standard libs -> third-party -> tactics2d."
                    )
                seen_project = True

    # --- Check pytest markers ---
    if markers_required and test_functions:
        has_markers = any("@pytest.mark." in line for line in lines)
        if not has_markers:
            markers_str = ", ".join(sorted(common_markers))
            warnings.append(
                "All test functions must have pytest markers "
                "(e.g., @pytest.mark.env). "
                f"Common markers: {markers_str}"
            )

    # --- Check for fixture docstrings ---
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("test_"):
                continue
            # Check if used as fixture (has fixture decorator)
            has_fixture_dec = any(
                isinstance(d, ast.Name)
                and d.id == "fixture"
                or isinstance(d, ast.Attribute)
                and d.attr == "fixture"
                for d in node.decorator_list
            )
            if has_fixture_dec and not ast.get_docstring(node):
                fixture_doc = doc_cfg.get("fixtureDocstring", "")
                if fixture_doc:
                    suggested = fixture_doc.replace("{resource}", node.name.replace("_", " "))
                    warnings.append(
                        f"Fixture '{node.name}' missing docstring. "
                        f'Expected something like: """{suggested}"""'
                    )
                else:
                    warnings.append(f"Fixture '{node.name}' missing docstring.")

    return warnings


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main():
    try:
        input_data = sys.stdin.read()
        if not input_data:
            return

        data = json.loads(input_data)
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        if tool_name not in ("Write", "Edit"):
            return

        file_path = tool_input.get("file_path", "")
        if not file_path or not _is_test_file(file_path):
            return

        rule = load_rule("pytest_style")
        if rule is None:
            print("Warning: pytest_style.json not found; skipping style check.", file=sys.stderr)
            return

        warnings = analyze_test_file(file_path, rule)
        if warnings:
            print(f"Pytest style suggestions for {file_path}:", file=sys.stderr)
            for w in warnings:
                print(f"  - {w}", file=sys.stderr)
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error in pytest style hook: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
