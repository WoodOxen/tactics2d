#!/usr/bin/env python3
"""Extract version from pyproject.toml (compatible with Python 3.8+)"""
import sys

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        # Fallback for minimal environments
        import json
        import re

        with open("pyproject.toml") as f:
            content = f.read()
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if match:
                print(match.group(1))
            else:
                print("Error: Could not find version in pyproject.toml", file=sys.stderr)
                sys.exit(1)
        sys.exit(0)

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)
    print(data["project"]["version"])
