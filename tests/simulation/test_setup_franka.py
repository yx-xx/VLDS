import os
from pathlib import Path
import shutil
import subprocess

import pytest

from openpi.simulation.paths import ROOT


@pytest.mark.parametrize("failures", [0, 1, 3])
def test_source_fetch_retries(tmp_path: Path, failures: int):
    repo = tmp_path / "repo"
    example = repo / "examples/libero"
    (example / ".cache/LIBERO/.git").mkdir(parents=True)
    (example / ".venv/bin").mkdir(parents=True)
    python = example / ".venv/bin/python"
    python.touch(mode=0o755)
    script = example / "setup_franka.sh"
    shutil.copyfile(ROOT / "examples/libero/setup_franka.sh", script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$GIT_CALLS"
if [[ "$*" == *"rev-parse --verify HEAD"* ]]; then
    exit 1
fi
if [[ "$*" == *"fetch --progress"* ]]; then
    ATTEMPTS=$(wc -l < "$FETCH_ATTEMPTS")
    printf 'attempt\\n' >> "$FETCH_ATTEMPTS"
    [[ "$ATTEMPTS" -ge "$FAILURES" ]]
fi
"""
    )
    git.chmod(0o755)
    for command in ("uv", "sleep"):
        executable = bin_dir / command
        executable.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$TOOL_CALLS"\nexit 0\n')
        executable.chmod(0o755)
    git_calls = tmp_path / "git_calls"
    fetch_attempts = tmp_path / "fetch_attempts"
    tool_calls = tmp_path / "tool_calls"
    fetch_attempts.touch()
    result = subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GIT_CALLS": str(git_calls),
            "FETCH_ATTEMPTS": str(fetch_attempts),
            "FAILURES": str(failures),
            "TOOL_CALLS": str(tool_calls),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    calls = git_calls.read_text().splitlines()
    fetches = [call for call in calls if "fetch --progress" in call]
    assert len(fetches) == min(failures + 1, 3)
    assert all("-c http.version=HTTP/1.1" in call for call in fetches)
    assert any("checkout --detach FETCH_HEAD" in call for call in calls) == (failures < 3)
    assert ("--config-setting editable_mode=compat" in tool_calls.read_text()) == (failures < 3)
    assert ("--reinstall-package libero" in tool_calls.read_text()) == (failures < 3)
    assert result.returncode == (1 if failures == 3 else 0), result.stderr
