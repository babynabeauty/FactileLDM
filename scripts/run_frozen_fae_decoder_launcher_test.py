"""Verify that the launcher uses only the requested checkout's environment."""

import os
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parent / "run_frozen_fae_decoder.sh"


def test_launcher_uses_project_python_and_drops_inherited_pythonpath(tmp_path):
    root = tmp_path / "FactileLDM"
    python = root / "env/.venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/sh\nprintf "using=%s\npath=%s\nhome=%s\nargs=%s\n" "$0" "$PYTHONPATH" "${PYTHONHOME-unset}" "$*"\n'
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "REPO_ROOT": str(root),
        "NUM_DEVICES": "4",
        "EXTRACT_BATCH_SIZE": "8",
        "PYTHON_BIN": "/wrong/venv/bin/python",
        "PYTHONPATH": "/wrong/venv/lib/site-packages",
        "PYTHONHOME": "/wrong/venv",
    }
    result = subprocess.run(["bash", str(SCRIPT), "prepare"], env=env, text=True, capture_output=True, check=True)
    assert f"using={python}" in result.stdout
    assert f"path={root}/src:{root}" in result.stdout
    assert "home=unset" in result.stdout
    assert "/wrong/venv" not in result.stdout

    smoke = subprocess.run(["bash", str(SCRIPT), "check"], env=env, text=True, capture_output=True, check=True)
    assert "--smoke --splits train" in smoke.stdout
    assert "--num-devices 4" in smoke.stdout
    assert "--batch-size 8" in smoke.stdout

    training = subprocess.run(["bash", str(SCRIPT), "train"], env=env, text=True, capture_output=True, check=True)
    assert "--num-devices 4" in training.stdout


def test_missing_project_python_fails_without_falling_back_to_path(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "prepare"],
        env={**os.environ, "REPO_ROOT": str(tmp_path)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "FactileLDM Python not found" in result.stderr
