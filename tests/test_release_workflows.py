# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute release workflow checks with real jq and offline GitHub responses."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="Workflow tests require PyYAML.")
ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


def run_step(workflow, name, releases, tmp_path):
    jq = os.environ.get("RELEASE_TEST_JQ") or shutil.which("jq")
    if not jq or not shutil.which("bash"):
        pytest.skip("Workflow execution tests require bash and jq.")
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    step = next(s for s in document["jobs"]["publish"]["steps"] if s["name"] == name)
    substitutions = {
        "steps.get_version.outputs.version": "0.1.9",
        "steps.create_draft_release.outputs.tag_name": "untagged-browser-id",
        "steps.draft_release.outputs.tag": "v0.1.9",
        "github.repository": "example/tactics2d",
    }
    script = re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda m: substitutions[m[1]], step["run"])
    output = tmp_path / "output"
    env = dict(
        os.environ,
        GITHUB_SHA=SHA,
        GITHUB_REPOSITORY="example/tactics2d",
        GITHUB_OUTPUT=str(output),
        RELEASE_TEST_JQ=jq,
        RELEASE_RESPONSE=json.dumps(releases),
        BASH_ENV="/dev/null",
    )
    # GitHub is mocked; jq and the workflow's entire shell block execute unchanged.
    prelude = """
jq() { "$RELEASE_TEST_JQ" "$@"; }
gh() {
    if [[ "$1" == api ]]; then
        printf '%s' "$RELEASE_RESPONSE"
    elif [[ "$1 $2" == 'release edit' ]]; then
        [[ "$3" == v0.1.9 ]] || return 90
        [[ "$*" == *"--tag v0.1.9 --target $GITHUB_SHA"* ]] || return 91
    else
        return 92
    fi
}
"""
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", prelude + script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, output.read_text() if output.exists() else ""


def draft():
    return {
        "draft": True,
        "name": "Release 0.1.9 (Test)",
        "tag_name": "v0.1.9",
        "target_commitish": SHA,
        "created_at": "2026-09-25T00:00:00Z",
        "assets": [{"name": f"wheel-{i}.whl"} for i in range(12)]
        + [{"name": "tactics2d-0.1.9.tar.gz"}],
    }


@pytest.mark.parametrize("case", ["valid", "missing_wheel", "wrong_sha", "published"])
def test_verify_draft_artifacts(case, tmp_path):
    release = draft()
    if case == "missing_wheel":
        release["assets"].pop(0)
    elif case == "wrong_sha":
        release["target_commitish"] = "b" * 40
    elif case == "published":
        release["draft"] = False
    result, _ = run_step(
        "publish_to_test_pypi.yml", "Verify Draft Release artifacts", [release], tmp_path
    )
    assert (result.returncode == 0) == (case == "valid"), result.stdout + result.stderr
    assert "compile error" not in result.stderr


def test_production_selects_current_draft(tmp_path):
    stale = dict(draft(), target_commitish="b" * 40)
    empty = dict(draft(), assets=[])
    result, output = run_step(
        "publish_to_pypi.yml",
        "Locate the validated draft release",
        [stale, empty, draft()],
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output == "tag=v0.1.9\n"


def test_publish_release_variables(tmp_path):
    result, _ = run_step("publish_to_pypi.yml", "Publish GitHub Release", [], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
