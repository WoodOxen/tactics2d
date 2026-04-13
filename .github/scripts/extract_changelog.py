#!/usr/bin/env python3
import re
import sys
from typing import Dict, List, Optional


class ReleaseChangelogExtractor:
    """Extract specific version changes from CHANGELOG.md"""

    def __init__(self, changelog_path: str = "CHANGELOG.md"):
        self.changelog_path = changelog_path
        self.content = ""

    def load_changelog(self) -> bool:
        """Load CHANGELOG.md content"""
        try:
            with open(self.changelog_path, encoding="utf-8") as f:
                self.content = f.read()
            return True
        except Exception:
            return False

    def extract_version_changes(self, version: str) -> Optional[Dict[str, List[str]]]:
        """Extract changes for a specific version"""
        if not self.content:
            return None

        # Pattern to find version section
        # Matches "## [0.1.9rc1] - 2026-01-09" or "## [0.1.9rc1]"
        version_pattern = rf"^##\s+\[{re.escape(version)}\](?:\s+-\s+\d{{4}}-\d{{2}}-\d{{2}})?"

        lines = self.content.split("\n")
        in_target_version = False
        current_section = ""
        changes = {}

        for line in lines:
            line_stripped = line.strip()

            # Check if we found the target version
            if re.match(version_pattern, line_stripped):
                in_target_version = True
                continue

            # If we're in the target version, check for section headers
            if in_target_version:
                # Check for end of version section (next version or end of file)
                if line_stripped.startswith("## ["):
                    break

                # Check for section header
                section_match = re.match(
                    r"^###\s+(Added|Changed|Fixed|Removed)", line_stripped, re.IGNORECASE
                )
                if section_match:
                    current_section = section_match.group(1).lower()
                    changes[current_section] = []
                    continue

                # Check for change entry
                if current_section and line_stripped.startswith("- "):
                    entry = line_stripped[2:].strip()
                    if entry:
                        changes[current_section].append(entry)

        return changes if changes else None

    def format_release_notes(self, version: str, changes: Dict[str, List[str]]) -> str:
        """Format changes for GitHub Release notes"""
        if not changes:
            return f"Release {version}"

        sections_order = ["added", "changed", "fixed", "removed"]
        output = [f"# Release {version}\n"]

        for section in sections_order:
            if section in changes and changes[section]:
                section_title = section.capitalize()
                output.append(f"## {section_title}")
                for entry in changes[section]:
                    output.append(f"- {entry}")
                output.append("")

        return "\n".join(output).strip()


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_changelog.py <version>")
        sys.exit(1)

    version = sys.argv[1]
    extractor = ReleaseChangelogExtractor()

    if not extractor.load_changelog():
        print(f"Error: Could not load CHANGELOG.md")
        sys.exit(1)

    changes = extractor.extract_version_changes(version)
    if changes is None:
        print(f"Warning: No changes found for version {version} in CHANGELOG.md")
        # Create minimal release notes
        release_notes = f"Release {version}"
    else:
        release_notes = extractor.format_release_notes(version, changes)

    # Write release notes to file for GitHub Actions
    with open("release_notes.txt", "w", encoding="utf-8") as f:
        f.write(release_notes)

    print(f"Extracted release notes for version {version}")


if __name__ == "__main__":
    main()
