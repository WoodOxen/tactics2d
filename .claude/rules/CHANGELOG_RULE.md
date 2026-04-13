# CHANGELOG Rule for Tactics2D

## Overview

This rule enforces CHANGELOG.md format compliance and provides automated PR description generation for the Tactics2D project. It helps maintain consistent release notes and streamlines the PR creation process.

## Rule Configuration

**File**: `.claude/rules/changelog_check.json`

**Key Features**:
1. **Format Validation**: Ensures CHANGELOG.md follows "Keep a Changelog" standard
2. **PR Description Extraction**: Automatically extracts changes from `[Unreleased]` section
3. **Warning System**: Provides suggestions without blocking workflow
4. **Integration**: Works with both file edits and PR creation commands

## Expected CHANGELOG Format

The rule expects CHANGELOG.md to follow this structure:

```markdown
# Change Log

## [Unreleased]

### Added
- New feature description

### Changed
- Change description

### Fixed
- Bug fix description

### Removed
- Removal description

## [0.1.9] - 2026-01-09

### Added
- Previous release features

### Changed
- Previous release changes
```

**Format Requirements**:
1. Must have `## [Unreleased]` section
2. Version headers: `## [version] - YYYY-MM-DD`
3. Section headers: `### Added`, `### Changed`, `### Fixed`, `### Removed`
4. Change entries start with `- ` (hyphen space)

## How It Works

### 1. File Edit Checks
When you edit `CHANGELOG.md` using Write or Edit tools:
- Hook runs automatically via PostToolUse
- Validates format compliance
- Shows warnings for issues (does not block)
- Suggestions appear in stderr

### 2. PR Creation Checks
When you run `gh pr create` or similar commands:
- Hook detects PR creation commands
- Parses `[Unreleased]` section
- Shows formatted changes for PR description
- Provides copy-paste ready output

## Usage Examples

### Example 1: Editing CHANGELOG.md
When you add changes to `[Unreleased]` section:

```markdown
## [Unreleased]

### Added
- New search algorithm implementation
- Tests for search algorithms

### Fixed
- Bug in geometry calculations
```

**Hook Output**:
```
CHANGELOG format suggestions for CHANGELOG.md:
  - [Unreleased] section appears empty. Consider adding changes.
Please follow 'Keep a Changelog' format.
```

### Example 2: Creating a PR
When you run `gh pr create --title "Update search algorithms"`:

**Hook Output**:
```
================================================================================
CHANGELOG.md [Unreleased] changes detected:
================================================================================

Suggested PR summary: Update: 2 addition(s), 1 fix(es)

Suggested PR description (copy and paste):
--------------------------------------------------------------------------------
### Added
- New search algorithm implementation
- Tests for search algorithms

### Fixed
- Bug in geometry calculations
--------------------------------------------------------------------------------

To use this in PR creation, add: --body "$(cat <<'EOF'
### Added
- New search algorithm implementation
- Tests for search algorithms

### Fixed
- Bug in geometry calculations
EOF
)"
================================================================================
```

## Integration with GitHub Actions

The local rule complements existing GitHub Actions:

| Component | Purpose | When It Runs |
|-----------|---------|--------------|
| **Local Hook** | Format validation & PR description | During local development |
| **sync_changelog.yml** | CHANGELOG synchronization | PR merged to main branch |

**Workflow**:
1. Developer updates `CHANGELOG.md` locally
2. Hook validates format and warns about issues
3. Developer creates PR with suggested description
4. GitHub Actions syncs changes after merge

## Configuration Details

### Hook Script: `check_changelog.py`
- **Location**: `.claude/hooks/check_changelog.py`
- **Triggers**: `Write|Edit` (CHANGELOG.md), `Bash` (PR commands)
- **Output**: Warning messages and PR description suggestions

### Settings Configuration
Added to `.claude/settings.json`:
```json
{
  "matcher": "Write|Edit",
  "hooks": [
    {
      "type": "command",
      "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/check_changelog.py"
    }
  ]
},
{
  "matcher": "Bash",
  "hooks": [
    {
      "type": "command",
      "command": "python3 \"$CLAUDE_PROJECT_DIR\"/.claude/hooks/check_changelog.py"
    }
  ]
}
```

## Troubleshooting

### Common Issues

#### 1. Hook Not Running
- Check file permissions: `chmod +x .claude/hooks/check_changelog.py`
- Verify settings.json configuration
- Ensure CHANGELOG.md filename is exact (case-sensitive)

#### 2. Format Warnings
- **Missing [Unreleased]**: Add `## [Unreleased]` section
- **Invalid section header**: Use only `### Added/Changed/Fixed/Removed`
- **Missing version date**: Format: `[0.1.9] - 2026-01-09`
- **Empty entries**: Remove or fill empty `- ` lines

#### 3. No PR Description Generated
- Ensure changes are in `[Unreleased]` section
- Check entry format: `- Description` (hyphen space)
- Verify hook is detecting `gh pr create` command

### Debugging

Test the hook manually:
```bash
# Test format validation
echo '{"tool_name":"Write","tool_input":{"file_path":"CHANGELOG.md"}}' | \
python3 .claude/hooks/check_changelog.py

# Test PR description extraction
echo '{"tool_name":"Bash","tool_input":{"command":"gh pr create --title \\"Test\\""}}' | \
python3 .claude/hooks/check_changelog.py
```

## Best Practices

### For Developers
1. **Update CHANGELOG First**: Add changes to `[Unreleased]` before coding
2. **Follow Format**: Use exact headers and bullet syntax
3. **Review Warnings**: Address format issues promptly
4. **Use Suggestions**: Copy PR description from hook output

### For Maintainers
1. **Keep [Unreleased] Clean**: Move to version section after release
2. **Regular Validation**: Run hook manually if unsure
3. **Update Documentation**: Keep this guide current

## Customization

### Modifying the Rule
Edit `.claude/rules/changelog_check.json`:
- `filePatterns`: Change target filename(s)
- `format`: Adjust format requirements
- `actions`: Modify trigger behavior
- `prDescriptionTemplate`: Customize PR output

### Extending Functionality
The hook script can be extended for:
- Additional format validations
- Integration with other tools
- Custom output formats
- Team-specific workflows

## Related Resources

- [Keep a Changelog](https://keepachangelog.com/) - Format standard
- [GitHub Actions: sync_changelog.yml](../.github/workflows/sync_changelog.yml)
- [Python Header Rule](python_header.json) - Similar rule pattern
- [Pytest Style Rule](pytest_style.json) - Similar rule pattern

## Changelog for This Rule

### [1.0] - 2026-01-23
- Initial implementation
- Basic format validation
- PR description extraction
- Integration with existing hook system

---

**Note**: This rule is designed to assist, not enforce. Warnings are suggestions, not requirements. The goal is to improve consistency and reduce manual work.
