#!/usr/bin/env python3
"""
Python file header checker for Tactics2D project.

Reads formatting rules from .claude/rules/python_header.json and
validates Python files after Write/Edit operations.
"""

import json
import os
import re
import sys
from typing import Optional

from rule_loader import load_rule

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _get_year() -> str:
    """Return the current year as a string."""
    # Use datetime only when actually called to avoid side-effects at import.
    import datetime

    return str(datetime.date.today().year)


def _match_pattern(filename: str, patterns: list) -> Optional[str]:
    """Return the first matching description template for *filename*."""
    for entry in patterns:
        pat = entry.get("pattern", "")
        template = entry.get("template", "")
        # Convert glob-like patterns to regex.
        regex = fnmatch_to_regex(pat)
        if re.search(regex, filename):
            return template
    return None


def fnmatch_to_regex(pat: str) -> str:
    """Convert a simple glob pattern to a regex string.

    Supports ``*`` (any sequence), and literal paths.
    """
    parts = re.split(r"(\\\*|\*)", pat)
    regex_parts = []
    for p in parts:
        if p == "*":
            regex_parts.append(r"[^/]*")
        elif p == "\\*":
            regex_parts.append(r"\*")
        else:
            regex_parts.append(re.escape(p))
    return "".join(regex_parts)


# --------------------------------------------------------------------------
# Validation logic
# --------------------------------------------------------------------------


def check_python_file(file_path: str, rule: dict) -> list:
    """Check a Python file for header compliance using *rule*.

    Returns a list of warning strings (empty if compliant).
    """
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            lines = [f.readline() for _ in range(12)]
    except OSError:
        return []

    header_cfg = rule.get("header", {})
    expected_copyright = header_cfg.get("copyright", "")
    expected_license = header_cfg.get("license", "")
    desc_rules = rule.get("descriptionRules", {})
    general = desc_rules.get("general", {})
    patterns = desc_rules.get("patterns", [])

    warnings = []

    # --- Copyright ---
    if expected_copyright:
        # Accept any year in the copyright line.
        copyright_template = expected_copyright.replace("{year}", r"\d{4}")
        copyright_found = any(re.search(copyright_template, line) for line in lines)
        if not copyright_found:
            year = _get_year()
            warnings.append(
                f"Missing or incorrect copyright notice. "
                f"Expected something like: "
                f'{expected_copyright.replace("{year}", year)}'
            )

    # --- SPDX ---
    if expected_license:
        spdx_found = any(expected_license in line for line in lines)
        if not spdx_found:
            warnings.append(f"Missing SPDX license identifier. " f"Add: {expected_license}")

    # --- Module docstring ---
    docstring_found = any(line.strip().startswith('"""') for line in lines)
    if not docstring_found:
        # Try to infer the expected description from the filename.
        basename = os.path.basename(file_path)
        template = _match_pattern(basename, patterns) or patterns[-1].get(
            "template", "{function_name} implementation."
        )
        warnings.append(
            f"Missing module docstring. "
            f"Expected a triple-quoted description like: "
            f'"""{template}"""'
        )

    # --- Description quality checks (if a docstring exists) ---
    docstring_line = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""'):
            docstring_line = stripped.strip('"')
            break

    if docstring_line:
        max_words = general.get("maxWords", 10)
        word_count = len(docstring_line.split())
        if word_count > max_words:
            warnings.append(
                f"Module docstring is {word_count} words " f"(max {max_words}). Keep it brief."
            )

        if general.get("endWithPeriod", True) and not docstring_line.endswith("."):
            warnings.append("Module docstring should end with a period.")

        if general.get("capitalizeFirst", True) and docstring_line:
            first_char = docstring_line[0]
            if first_char.isalpha() and not first_char.isupper():
                warnings.append("Module docstring should start with a capital letter.")

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
        if not file_path or not file_path.endswith(".py"):
            return

        rule = load_rule("python_header")
        if rule is None:
            print("Warning: python_header.json not found; skipping header check.", file=sys.stderr)
            return

        warnings = check_python_file(file_path, rule)
        if warnings:
            print(f"Warning: Python file header issues in {file_path}:", file=sys.stderr)
            for w in warnings:
                print(f"  - {w}", file=sys.stderr)
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error in python header hook: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
