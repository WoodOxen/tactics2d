#!/usr/bin/env python3
"""
Shared rule loader for Tactics2D hook scripts.

Loads structured rules from .claude/rules/*.json so that hook scripts
are driven by the rule definitions rather than hardcoded logic.
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional


def _get_rules_dir() -> str:
    """Return the absolute path to .claude/rules/.

    Resolves via CLAUDE_PROJECT_DIR first, then falls back to a
    heuristic based on this script's location.
    """
    env_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_dir:
        candidate = os.path.join(env_dir, ".claude", "rules")
        if os.path.isdir(candidate):
            return candidate

    # Fallback: resolve relative to this script's location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "..", "rules")
    if os.path.isdir(candidate):
        return os.path.normpath(candidate)

    raise FileNotFoundError(
        "Cannot locate .claude/rules/ directory. "
        "Set CLAUDE_PROJECT_DIR or ensure hooks live under .claude/hooks/"
    )


@lru_cache(maxsize=8)
def load_rule(rule_name: str) -> Optional[Dict[str, Any]]:
    """Load a rule JSON file by its base name (with or without .json suffix).

    Returns the parsed dict, or None if the file does not exist or is invalid.
    """
    if not rule_name.endswith(".json"):
        rule_name = f"{rule_name}.json"

    rules_dir = _get_rules_dir()
    path = os.path.join(rules_dir, rule_name)

    if not os.path.isfile(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_project_dir() -> str:
    """Return the project root directory."""
    env_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_dir:
        return env_dir

    # Fallback: parent of .claude/
    rules_dir = _get_rules_dir()
    return os.path.normpath(os.path.join(rules_dir, "..", ".."))


def get_hooks_dir() -> str:
    """Return the absolute path to .claude/hooks/."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "hooks":
        return script_dir
    # Called from outside .claude/hooks/ — resolve via project dir.
    return os.path.join(get_project_dir(), ".claude", "hooks")
