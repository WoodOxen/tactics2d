#!/usr/bin/env python3
import fnmatch
import os
import sys
import tarfile
from pathlib import Path


def create_source_archive(version: str):
    """Create source archive for GitHub Release"""
    archive_name = f"tactics2d-{version}.tar.gz"

    # Files to exclude from source archive
    exclude_patterns = [
        ".git",
        ".github",
        ".claude",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "build",
        "dist",
        "final_dist",
        "*.egg-info",
        "tests/runtime",
        "tactics2d/data",
        "*.so",
        "*.so.*",
        "release_notes.txt",
        "tactics2d-*.tar.gz",
    ]

    with tarfile.open(archive_name, "w:gz") as tar:
        for root, dirs, files in os.walk("."):
            # Skip excluded directories using fnmatch
            dirs[:] = [
                d
                for d in dirs
                if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns)
            ]

            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, ".")

                # Skip excluded files using fnmatch
                if any(fnmatch.fnmatch(arcname, pattern) for pattern in exclude_patterns):
                    continue

                # Skip if any part of the path matches exclude pattern
                skip = False
                for part in Path(arcname).parts:
                    if any(fnmatch.fnmatch(part, pattern) for pattern in exclude_patterns):
                        skip = True
                        break
                if skip:
                    continue

                try:
                    tar.add(file_path, arcname=arcname)
                except Exception as e:
                    print(f"Warning: Could not add {file_path}: {e}")

    print(f"Created source archive: {archive_name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_source_archive.py <version>")
        sys.exit(1)

    version = sys.argv[1]
    create_source_archive(version)
