#!/usr/bin/env python3
"""Check if a version exists on PyPI or Test PyPI.

Usage:
  python check_pypi_version.py <version> [--test] [--repository-url URL]
  python check_pypi_version.py <version> --repository-url https://test.pypi.org/legacy/

Exit codes:
  0: Version exists on the specified repository
  1: Version does not exist on the specified repository
  2: Error occurred (network issue, invalid arguments, etc.)
"""

import json
import sys
import urllib.error
import urllib.request
from argparse import ArgumentParser


def version_exists(version: str, repository_url: str = "https://pypi.org") -> bool:
    """Check if the given version exists in the PyPI repository.

    Args:
        version: Package version to check (e.g., "0.1.9rc3")
        repository_url: Base URL of the PyPI repository

    Returns:
        True if version exists, False otherwise
    """
    package_name = "tactics2d"

    # Normalize repository URL
    if repository_url.endswith("/legacy/"):
        # For compatibility with --repository-url https://test.pypi.org/legacy/
        base_url = repository_url.replace("/legacy/", "")
    elif repository_url.endswith("/"):
        base_url = repository_url.rstrip("/")
    else:
        base_url = repository_url

    # Construct the JSON API URL
    api_url = f"{base_url}/pypi/{package_name}/json"

    try:
        # Fetch package metadata from PyPI
        with urllib.request.urlopen(api_url, timeout=10) as response:
            if response.status != 200:
                print(f"Error: HTTP {response.status} from {api_url}", file=sys.stderr)
                return False

            data = json.load(response)
            versions = data.get("releases", {})

            # Check if the version exists in the releases
            exists = version in versions
            if exists:
                print(f"Version {version} exists in {repository_url}")
            else:
                print(f"Version {version} does not exist in {repository_url}")

            return exists

    except urllib.error.URLError as e:
        print(f"Network error checking {api_url}: {e}", file=sys.stderr)
        # If we can't connect, assume version doesn't exist to be safe
        return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response from {api_url}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return False


def main():
    parser = ArgumentParser(description="Check if a version exists on PyPI")
    parser.add_argument("version", help="Package version to check")
    parser.add_argument("--test", action="store_true", help="Check Test PyPI instead of PyPI")
    parser.add_argument(
        "--repository-url", help="Custom repository URL (e.g., https://test.pypi.org/legacy/)"
    )

    args = parser.parse_args()

    # Determine which repository to check
    if args.repository_url:
        repository_url = args.repository_url
    elif args.test:
        repository_url = "https://test.pypi.org"
    else:
        repository_url = "https://pypi.org"

    try:
        exists = version_exists(args.version, repository_url)
        sys.exit(0 if exists else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
