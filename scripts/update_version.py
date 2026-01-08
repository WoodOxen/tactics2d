##! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: update_version.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.8rc1

"""
Version updater for tactics2d.

This script updates version information across the codebase:
1. Updates all @Version: tags in Python file headers
2. Updates tactics2d/__init__.py __version__
3. Ensures consistency with pyproject.toml

Usage:
    python scripts/update_version.py [--dry-run] [--version VERSION]

If --version is not specified, reads version from pyproject.toml.
"""

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def read_version_from_pyproject():
    """Read version from pyproject.toml using regex."""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    content = pyproject_path.read_text(encoding="utf-8")
    # Match version = "x.y.z" (with optional release candidate suffix)
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {pyproject_path}")
    return match.group(1)


def update_file_version_tags(version, dry_run=False):
    """Update @Version: tags in all Python files."""
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip hidden directories and virtual environments
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in ("__pycache__", "venv", "env", ".venv")
        ]
        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)

    updated_files = []
    pattern = re.compile(r"^(\s*#\s*@Version:\s*).*$", re.MULTILINE)

    for filepath in python_files:
        content = filepath.read_text(encoding="utf-8")
        new_content, count = pattern.subn(rf"\g<1>{version}", content)
        if count > 0:
            updated_files.append((filepath, count))
            if not dry_run:
                filepath.write_text(new_content, encoding="utf-8")

    return updated_files


def update_init_version(version, dry_run=False):
    """Update __version__ in tactics2d/__init__.py."""
    init_file = PROJECT_ROOT / "tactics2d" / "__init__.py"
    content = init_file.read_text(encoding="utf-8")

    # Update __version__ assignment
    pattern = re.compile(r'^(__version__\s*=\s*").*(")$', re.MULTILINE)
    new_content, count = pattern.subn(rf"\g<1>{version}\2", content)

    if count == 0:
        # Try alternative pattern
        pattern2 = re.compile(r"^(__version__\s*=\s*)[^\n]+$", re.MULTILINE)
        new_content, count = pattern2.subn(rf'\g<1>"{version}"', content)

    if count > 0:
        if not dry_run:
            init_file.write_text(new_content, encoding="utf-8")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Update version information across the codebase")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be changed without modifying files"
    )
    parser.add_argument("--version", help="Version to use (default: read from pyproject.toml)")
    args = parser.parse_args()

    try:
        if args.version:
            version = args.version
        else:
            version = read_version_from_pyproject()

        print(f"Using version: {version}")
        print(f"Dry run: {args.dry_run}")

        # Update __init__.py
        if update_init_version(version, args.dry_run):
            print(f"✓ Updated tactics2d/__init__.py __version__ to {version}")
        else:
            print("✗ Could not find __version__ in tactics2d/__init__.py")

        # Update @Version tags
        updated = update_file_version_tags(version, args.dry_run)
        if updated:
            print(f"✓ Updated @Version tags in {len(updated)} files:")
            for filepath, count in updated:
                print(
                    f"  - {filepath.relative_to(PROJECT_ROOT)} ({count} tag{'s' if count > 1 else ''})"
                )
        else:
            print("✗ No @Version tags found to update")

        if args.dry_run:
            print("\nDry run completed. No files were modified.")
        else:
            print("\nVersion update completed successfully.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
