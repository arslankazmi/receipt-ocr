"""Tests for training pipeline — uses --dry-run to avoid full downloads."""
import subprocess
import sys
from pathlib import Path

import pytest

CONFIG = Path("config.yaml")


@pytest.mark.skipif(not CONFIG.exists(), reason="config.yaml not found")
def test_finetune_dry_run():
    """Verify training script imports, loads model, runs 1 batch, exits cleanly."""
    result = subprocess.run(
        [
            sys.executable,
            "train/finetune_donut.py",
            "--config",
            str(CONFIG),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=300,
    )
    assert result.returncode == 0, f"dry-run failed:\n{result.stderr}"
