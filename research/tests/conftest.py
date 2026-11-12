"""Shared fixtures: one tiny offline training run that the Phase 1 tests build on."""

from pathlib import Path

import pytest

import train_sae


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory) -> Path:
    """Train the offline test SAE once (random 4-layer GPT-2, synthetic text, a few seconds on CPU)."""
    out = tmp_path_factory.mktemp("runs")
    assert train_sae.main(["--random", "--tiny", "--out", str(out), "--name", "tiny", "--device", "cpu"]) == 0
    return out / "tiny"


@pytest.fixture(scope="session")
def tiny_examples(tiny_run) -> Path:
    """Top examples of the tiny SAE, collected once (``<run>/examples``)."""
    import collect_top

    assert collect_top.main(["--sae", str(tiny_run), "--random", "--tiny", "--device", "cpu"]) == 0
    return tiny_run / "examples"
