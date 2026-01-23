#!/usr/bin/env python3
"""
CHANGELOG.md checker for Tactics2D project.
This hook checks CHANGELOG.md format compliance and extracts changes for PR descriptions.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ChangelogParser:
    """Parser for CHANGELOG.md following 'Keep a Changelog' format."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content = ""
        self.sections: Dict[str, Dict[str, List[str]]] = {}
        self.current_version = ""
        self.current_section = ""
        self.has_unreleased = False
        self.unreleased_changes: Dict[str, List[str]] = {}

    def parse(self) -> bool:
        """Parse the CHANGELOG.md file."""
        if not os.path.exists(self.file_path):
            return False

        try:
            with open(self.file_path, encoding="utf-8") as f:
                self.content = f.read()
        except Exception:
            return False

        lines = self.content.split("\n")
        self.sections = {}
        self.unreleased_changes = {}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for version header
            version_match = re.match(r"^##\s+\[([^\]]+)\](?:\s+-\s+(\d{4}-\d{2}-\d{2}))?", line)
            if version_match:
                version = version_match.group(1)
                self.current_version = version
                self.sections[version] = {}
                self.has_unreleased = version == "Unreleased"
                i += 1
                continue

            # Check for section header within a version
            section_match = re.match(r"^###\s+(Added|Changed|Fixed|Removed)", line, re.IGNORECASE)
            if section_match and self.current_version:
                section = section_match.group(1).lower()
                self.current_section = section
                self.sections[self.current_version][section] = []
                i += 1
                continue

            # Check for change entry
            if self.current_version and self.current_section:
                entry_match = re.match(r"^-\s+(.+)", line)
                if entry_match:
                    entry = entry_match.group(1).strip()
                    if entry:  # Skip empty entries
                        self.sections[self.current_version][self.current_section].append(entry)

                        # Store unreleased changes separately
                        if self.current_version == "Unreleased":
                            if section not in self.unreleased_changes:
                                self.unreleased_changes[section] = []
                            self.unreleased_changes[section].append(entry)

            i += 1

        return True

    def validate_format(self) -> List[str]:
        """Validate CHANGELOG.md format and return warnings."""
        warnings = []

        if not self.content:
            warnings.append("CHANGELOG.md is empty")
            return warnings

        # Check for [Unreleased] section
        if "## [Unreleased]" not in self.content:
            warnings.append("Missing [Unreleased] section. Add '## [Unreleased]' header.")
        else:
            # Check if [Unreleased] has content
            if not self.unreleased_changes:
                warnings.append("[Unreleased] section appears empty. Consider adding changes.")

        # Check for proper version headers
        version_headers = re.findall(
            r"^##\s+\[([^\]]+)\](?:\s+-\s+(\d{4}-\d{2}-\d{2}))?", self.content, re.MULTILINE
        )
        for version, date in version_headers:
            if version != "Unreleased" and not date:
                warnings.append(
                    f"Version '{version}' missing date. Format: '[{version}] - YYYY-MM-DD'"
                )

        # Check section headers
        lines = self.content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check for malformed section headers
            if stripped.startswith("###") and not re.match(
                r"^###\s+(Added|Changed|Fixed|Removed)", stripped, re.IGNORECASE
            ):
                warnings.append(
                    f"Invalid section header: '{stripped}'. Use ### Added/Changed/Fixed/Removed"
                )

        return warnings

    def has_unreleased_changes(self) -> bool:
        """Check if [Unreleased] section has changes."""
        if not self.unreleased_changes:
            return False
        # Check if any section has entries
        for section_entries in self.unreleased_changes.values():
            if section_entries:
                return True
        return False

    def format_changes_for_pr(self) -> str:
        """Format [Unreleased] changes for PR description."""
        if not self.has_unreleased_changes():
            return "No changes found in [Unreleased] section."

        sections_order = ["added", "changed", "fixed", "removed"]
        output_lines = []

        for section in sections_order:
            if section in self.unreleased_changes and self.unreleased_changes[section]:
                # Capitalize section name
                section_title = section.capitalize()
                output_lines.append(f"### {section_title}")
                for entry in self.unreleased_changes[section]:
                    output_lines.append(f"- {entry}")
                output_lines.append("")  # Add blank line

        # Remove trailing blank line
        if output_lines and output_lines[-1] == "":
            output_lines.pop()

        return "\n".join(output_lines)

    def get_pr_description_summary(self) -> str:
        """Generate a summary of changes for PR title/description."""
        if not self.has_unreleased_changes():
            return "Update CHANGELOG.md"

        # Count changes by type
        counts = {}
        for section, entries in self.unreleased_changes.items():
            if entries:
                counts[section] = len(entries)

        # Build summary
        parts = []
        if "added" in counts:
            parts.append(f"{counts['added']} addition(s)")
        if "changed" in counts:
            parts.append(f"{counts['changed']} change(s)")
        if "fixed" in counts:
            parts.append(f"{counts['fixed']} fix(es)")
        if "removed" in counts:
            parts.append(f"{counts['removed']} removal(s)")

        if parts:
            return f"Update: {', '.join(parts)}"
        return "Update CHANGELOG.md"


def check_changelog_format(file_path: str) -> List[str]:
    """Check CHANGELOG.md file for format compliance."""
    parser = ChangelogParser(file_path)
    if not parser.parse():
        return ["Failed to parse CHANGELOG.md"]

    return parser.validate_format()


def extract_pr_description(file_path: str) -> Tuple[str, str]:
    """Extract PR description from CHANGELOG.md [Unreleased] section."""
    parser = ChangelogParser(file_path)
    if not parser.parse():
        return "Update CHANGELOG.md", "Failed to parse CHANGELOG.md"

    summary = parser.get_pr_description_summary()
    changes = parser.format_changes_for_pr()

    return summary, changes


def check_pr_command(tool_input: Dict) -> bool:
    """Check if the bash command is a PR creation command."""
    command = tool_input.get("command", "")
    return "gh pr create" in command or "git push" in command and "origin" in command


def main():
    """Main hook function."""
    # Read hook input from stdin
    try:
        input_data = sys.stdin.read()
        if not input_data:
            return

        data = json.loads(input_data)
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        # Get project directory
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")
        changelog_path = os.path.join(project_dir, "CHANGELOG.md")

        # Handle Write/Edit operations on CHANGELOG.md
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            if file_path and os.path.basename(file_path) == "CHANGELOG.md":
                warnings = check_changelog_format(file_path)
                if warnings:
                    print(f"CHANGELOG format suggestions for {file_path}:", file=sys.stderr)
                    for warning in warnings:
                        print(f"  - {warning}", file=sys.stderr)
                    print("Please follow 'Keep a Changelog' format.", file=sys.stderr)

        # Handle Bash commands for PR creation
        elif tool_name == "Bash" and check_pr_command(tool_input):
            if not os.path.exists(changelog_path):
                print("Warning: CHANGELOG.md file not found.", file=sys.stderr)
                return

            parser = ChangelogParser(changelog_path)
            if not parser.parse():
                print("Warning: Failed to parse CHANGELOG.md.", file=sys.stderr)
                return

            # Check for unreleased changes
            if parser.has_unreleased_changes():
                summary = parser.get_pr_description_summary()
                changes = parser.format_changes_for_pr()

                print("\n" + "=" * 80, file=sys.stderr)
                print("CHANGELOG.md [Unreleased] changes detected:", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                print(f"\nSuggested PR summary: {summary}\n", file=sys.stderr)
                print("Suggested PR description (copy and paste):", file=sys.stderr)
                print("-" * 80, file=sys.stderr)
                print(changes, file=sys.stderr)
                print("-" * 80, file=sys.stderr)
                print("\nTo use this in PR creation, add: --body \"$(cat <<'EOF'", file=sys.stderr)
                print(changes, file=sys.stderr)
                print('EOF\n)"', file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)
            else:
                print("Warning: CHANGELOG.md [Unreleased] section appears empty.", file=sys.stderr)
                print("Consider adding changes before creating PR.", file=sys.stderr)

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error in CHANGELOG hook: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
