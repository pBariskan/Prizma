"""Shared fixture: the whole Phase 1 pipeline, run once on the offline test model."""

from pathlib import Path

import pytest

import run_phase1


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory) -> Path:
    """Train, collect examples, evaluate and label (offline stand-in) with ``run_phase1.py --random --tiny``.

    A random 4-layer GPT-2 on synthetic text: no downloads, about half a minute on a CPU.
    """
    out = tmp_path_factory.mktemp("runs")
    assert run_phase1.main(["--random", "--tiny", "--device", "cpu", "--out", str(out), "--name", "tiny"]) == 0
    return out / "tiny"
