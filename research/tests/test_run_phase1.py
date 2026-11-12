"""The one-command pipeline: outputs, skipping finished steps, continuing an interrupted training."""

import json

import pytest

import run_phase1
import train_sae


def test_pipeline_writes_every_output(tiny_run):
    for path in ("cfg.json", "sae_weights.safetensors", "train_summary.json", "examples/examples.safetensors",
                 "examples/meta.json", "eval.json", "eval.md", "autointerp/labels.jsonl", "autointerp/scores.jsonl",
                 "web/sae/labels.json", "web/sae/concepts/0000.json"):
        assert (tiny_run / path).exists(), path
    meta = json.loads((tiny_run / "examples" / "meta.json").read_text())
    assert meta["sae"] == "tiny" and meta["context_size"] == 32 and meta["mean_l0"] == pytest.approx(8, abs=0.5)


def test_finished_steps_are_not_run_again(tiny_run, monkeypatch, capsys):
    def boom(argv):
        raise AssertionError("a finished step ran again")

    for script in ("train_sae", "collect_top", "eval"):
        monkeypatch.setattr(__import__(script), "main", boom)
    assert run_phase1.main(["--random", "--tiny", "--device", "cpu", "--out", str(tiny_run.parent), "--name", "tiny"]) == 0
    out = capsys.readouterr().out
    assert out.count("already done") == 3 and "[label] nothing to send" in out


def test_a_failing_step_stops_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(train_sae, "main", lambda argv: 1)
    with pytest.raises(SystemExit, match="failed with exit code 1"):
        run_phase1.main(["--random", "--tiny", "--out", str(tmp_path)])
    assert not (tmp_path / "pipeline-test" / "examples").exists()


def test_real_run_starts_with_the_smoke_test(tmp_path, monkeypatch):
    """Without --random/--tiny: the four steps with --tiny into runs/smoke first, then the real ones."""
    calls = []
    for script in ("train_sae", "collect_top", "eval", "autointerp"):
        monkeypatch.setattr(__import__(script), "main", lambda argv, script=script: calls.append((script, argv)) or 0)
    monkeypatch.setattr(run_phase1.Path, "read_text", lambda self: "")
    assert run_phase1.main(["--out", str(tmp_path)]) == 0
    assert [script for script, _ in calls] == ["train_sae", "collect_top", "eval", "autointerp"] * 2
    assert all("--tiny" in argv for _, argv in calls[:3]) and not any("--tiny" in argv for _, argv in calls[4:])
    assert calls[0][1][-4:] == ["--name", "smoke", "--resume-from", "auto"]
    assert calls[4][1] == ["--out", str(tmp_path), "--name", run_phase1.FULL_NAME, "--resume-from", "auto"]
    assert calls[7][1] == ["--run", str(tmp_path / run_phase1.FULL_NAME)]  # dry run: no --submit, ever
    assert not any("--submit" in argv or "--fake" in argv for _, argv in calls)


def test_interrupted_training_continues_from_the_last_checkpoint(tmp_path):
    args = ["--random", "--tiny", "--device", "cpu", "--out", str(tmp_path), "--name", "run", "--checkpoints", "2",
            "--training-tokens", "150000"]
    assert train_sae.main(args) == 0
    checkpoint = train_sae.latest_checkpoint(tmp_path / "run")
    assert checkpoint is not None and int(checkpoint.name) == max(
        int(p.name) for p in (tmp_path / "run" / "checkpoints").glob("*/*"))
    first_log = [json.loads(line) for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]

    assert train_sae.main([*args, "--resume-from", "auto"]) == 0
    log = [json.loads(line) for line in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    steps = [row["step"] for row in log]
    assert steps == sorted(set(steps)) and steps[-1] == first_log[-1]["step"]  # one continuous log, same length
    resumed_at = int(checkpoint.name) // 1024
    first_new = next(row for row in log if row["step"] > resumed_at)
    assert first_new["explained_variance"] > 0.3  # it did not start from scratch
    assert train_sae.latest_checkpoint(tmp_path / "nothing-here") is None
