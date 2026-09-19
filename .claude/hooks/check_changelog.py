#!/usr/bin/env python3
"""
CHANGELOG.md checker for Tactics2D project.

Reads formatting rules from .claude/rules/changelog_check.json and
validates CHANGELOG.md format after Write/Edit operations, and
extracts PR descriptions when detecting ``gh pr create``.
"""

import json
import os
import re
import sys
from typing import Dict, List, Tuple

from rule_loader import get_project_dir, load_rule

# --------------------------------------------------------------------------
# ChangelogParser
# --------------------------------------------------------------------------


class ChangelogParser:
    """Parse and validate CHANGELOG.md driven by a rule definition."""

    def __init__(self, file_path: str, rule: dict):
        self.file_path = file_path
        self.rule = rule
        self.content = ""
        self.sections: Dict[str, Dict[str, List[str]]] = {}
        self.unreleased_changes: Dict[str, List[str]] = {}
        self._current_version = ""
        self._current_section = ""

    def parse(self) -> bool:
        """Parse the CHANGELOG.md file.

        Returns True on success (even if the file is empty).
        """
        if not os.path.exists(self.file_path):
            return False

        try:
            with open(self.file_path, encoding="utf-8") as f:
                self.content = f.read()
        except OSError:
            return False

        self.sections = {}
        self.unreleased_changes = {}
        lines = self.content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Version header: ## [version] or ## [version] - YYYY-MM-DD
            version_match = re.match(r"^##\s+\[([^\]]+)\](?:\s+-\s+(\d{4}-\d{2}-\d{2}))?", line)
            if version_match:
                self._current_version = version_match.group(1)
                self.sections[self._current_version] = {}
                i += 1
                continue

            # Section header: ### Added | Changed | Fixed | Removed
            section_match = re.match(r"^###\s+(Added|Changed|Fixed|Removed)", line, re.IGNORECASE)
            if section_match and self._current_version:
                self._current_section = section_match.group(1).lower()
                self.sections.setdefault(self._current_version, {})[self._current_section] = []
                i += 1
                continue

            # Entry: - description
            if self._current_version and self._current_section:
                entry_match = re.match(r"^-\s+(.+)", line)
                if entry_match:
                    entry = entry_match.group(1).strip()
                    if entry:
                        self.sections[self._current_version][self._current_section].append(entry)
                        if self._current_version == "Unreleased":
                            self.unreleased_changes.setdefault(self._current_section, []).append(
                                entry
                            )
            i += 1

        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_format(self) -> List[str]:
        """Validate CHANGELOG.md format against the rule definition."""
        warnings: List[str] = []
        fmt = self.rule.get("format", {})
        required_sections = fmt.get("requiredSections", ["[Unreleased]"])
        allowed_headers = fmt.get("sectionHeaders", [])
        entry_fmt = fmt.get("entryFormat", "- Item description")

        if not self.content:
            warnings.append("CHANGELOG.md is empty")
            return warnings

        # Required sections
        for req in required_sections:
            header = f"## [{req}]"
            if header not in self.content:
                warnings.append(f"Missing required section '{header}'. " f"Add '{header}' header.")
            elif req == "Unreleased" and not self.unreleased_changes:
                warnings.append("[Unreleased] section appears empty. Consider adding changes.")

        # Version dates
        version_headers = re.findall(
            r"^##\s+\[([^\]]+)\](?:\s+-\s+(\d{4}-\d{2}-\d{2}))?", self.content, re.MULTILINE
        )
        for version, date in version_headers:
            if version != "Unreleased" and not date:
                warnings.append(
                    f"Version '{version}' missing date. " f"Format: '[{version}] - YYYY-MM-DD'"
                )

        # Invalid section headers
        lines = self.content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("###"):
                # Check it matches an allowed header.
                allowed = False
                for hdr in allowed_headers:
                    if stripped.lower() == hdr.lower():
                        allowed = True
                        break
                if not allowed:
                    allowed_str = ", ".join(allowed_headers)
                    warnings.append(
                        f"Invalid section header: '{stripped}'. " f"Use one of: {allowed_str}"
                    )

        return warnings

    # ------------------------------------------------------------------
    # PR description helpers
    # ------------------------------------------------------------------

    def has_unreleased_changes(self) -> bool:
        """Return True if [Unreleased] has at least one entry."""
        return any(bool(entries) for entries in self.unreleased_changes.values())

    def format_changes_for_pr(self) -> str:
        """Format [Unreleased] changes as a PR description body."""
        if not self.has_unreleased_changes():
            return "No changes found in [Unreleased] section."

        sections_order = ["added", "changed", "fixed", "removed"]
        output_lines = []
        for section in sections_order:
            entries = self.unreleased_changes.get(section, [])
            if entries:
                output_lines.append(f"### {section.capitalize()}")
                for entry in entries:
                    output_lines.append(f"- {entry}")
                output_lines.append("")

        # Remove trailing blank line
        if output_lines and output_lines[-1] == "":
            output_lines.pop()

        return "\n".join(output_lines)

    def get_pr_description_summary(self) -> str:
        """Generate a one-line summary of the unreleased changes."""
        if not self.has_unreleased_changes():
            return "Update CHANGELOG.md"

        counts = {
            section: len(entries) for section, entries in self.unreleased_changes.items() if entries
        }
        parts = []
        labels = {
            "added": "addition(s)",
            "changed": "change(s)",
            "fixed": "fix(es)",
            "removed": "removal(s)",
        }
        for section, count in counts.items():
            label = labels.get(section, section)
            parts.append(f"{count} {label}")

        return f"Update: {', '.join(parts)}" if parts else "Update CHANGELOG.md"


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------


def check_changelog_format(file_path: str, rule: dict) -> List[str]:
    """Check CHANGELOG.md file for format compliance."""
    parser = ChangelogParser(file_path, rule)
    if not parser.parse():
        return ["Failed to parse CHANGELOG.md"]
    return parser.validate_format()


def extract_pr_description(file_path: str, rule: dict) -> Tuple[str, str]:
    """Extract PR description from [Unreleased] section.

    Returns (summary, body).
    """
    parser = ChangelogParser(file_path, rule)
    if not parser.parse():
        return "Update CHANGELOG.md", "Failed to parse CHANGELOG.md"
    return parser.get_pr_description_summary(), parser.format_changes_for_pr()


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

        rule = load_rule("changelog_check")
        if rule is None:
            print(
                "Warning: changelog_check.json not found; " "skipping CHANGELOG validation.",
                file=sys.stderr,
            )
            return

        project_dir = get_project_dir()
        changelog_path = os.path.join(project_dir, "CHANGELOG.md")

        # --- Write/Edit on CHANGELOG.md ---
        if tool_name in ("Write", "Edit"):
            file_path = tool_input.get("file_path", "")
            if file_path and os.path.basename(file_path) == "CHANGELOG.md":
                warnings = check_changelog_format(file_path, rule)
                if warnings:
                    print(f"CHANGELOG format suggestions for {file_path}:", file=sys.stderr)
                    for w in warnings:
                        print(f"  - {w}", file=sys.stderr)

        # --- Bash: PR creation commands ---
        elif tool_name == "Bash":
            command = tool_input.get("command", "")
            is_pr_cmd = "gh pr create" in command or ("git push" in command and "origin" in command)
            if not is_pr_cmd:
                return

            if not os.path.exists(changelog_path):
                print("Warning: CHANGELOG.md file not found.", file=sys.stderr)
                return

            summary, body = extract_pr_description(changelog_path, rule)
            if not body or body.startswith("No changes"):
                print(
                    "Warning: CHANGELOG.md [Unreleased] section "
                    "appears empty. Consider adding changes before "
                    "creating PR.",
                    file=sys.stderr,
                )
                return

            print("\n" + "=" * 80, file=sys.stderr)
            print("CHANGELOG.md [Unreleased] changes detected:", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"\nSuggested PR summary: {summary}\n", file=sys.stderr)
            print("Suggested PR description (copy and paste):", file=sys.stderr)
            print("-" * 80, file=sys.stderr)
            print(body, file=sys.stderr)
            print("-" * 80, file=sys.stderr)
            print("\nTo use this in PR creation, add: " "--body \"$(cat <<'EOF'", file=sys.stderr)
            print(body, file=sys.stderr)
            print("EOF\n)", file=sys.stderr)
            print("=" * 80 + "\n", file=sys.stderr)

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error in CHANGELOG hook: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
