"""Phase 1 in one command: train the SAE, collect examples, evaluate, and price the labeling.

    uv run python run_phase1.py                   # the real run, on a rented GPU (see docs/gpu-runbook.md)
    uv run python run_phase1.py --tiny            # everything with real GPT-2 but small budgets (minutes)
    uv run python run_phase1.py --random --tiny   # the whole pipeline offline on a CPU in about a minute

Runs, in order, into ``runs/<name>/``:

    train_sae.py      TopK SAE on blocks.6.hook_resid_pre       -> cfg.json, sae_weights.safetensors
    collect_top.py    strongest and random examples per concept  -> examples/
    eval.py           metrics next to the open SAE, gate check   -> eval.json, eval.md
    autointerp.py     dry run: what labeling would cost          (sends nothing)

The real run starts with a smoke test: the same four steps with small budgets into ``runs/smoke``.
It takes a few minutes and fails early if the model, the dataset or the open SAE cannot be
downloaded, or the GPU is not used. Steps that have finished are skipped when the command is run
again, and an interrupted training continues from its last checkpoint.

The only step that costs money beyond the GPU is the labeling itself, and this script never starts
it: run ``autointerp.py --run runs/<name> --submit`` yourself. With ``--random`` the labels come
from an offline stand-in, so that the export for the site is exercised too.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from prizma_research.common import RUNS_DIR

FULL_NAME = "gpt2-l6-topk16384-k32"


def step(title: str, script: str, argv: list[str], done: Path | None) -> None:
    """Run one script's ``main`` unless its output is already there."""
    command = f"uv run python {script}.py {shlex.join(argv)}"
    if done is not None and done.exists():
        print(f"\n=== {title}: already done ({done}), skipping ===", flush=True)
        return
    print(f"\n=== {title} ===\n$ {command}", flush=True)
    started = time.time()
    code = __import__(script).main(argv)
    if code != 0:
        raise SystemExit(f"{title} failed with exit code {code}. Fix the problem and run the same command again.")
    print(f"=== {title}: finished in {(time.time() - started) / 60:.1f} min ===", flush=True)


def pipeline(name: str, runs: Path, mode: list[str], *, fake_labels: bool, train_args: list[str]) -> Path:
    run = runs / name
    step(f"Train {name}", "train_sae", [*mode, "--out", str(runs), "--name", name, "--resume-from", "auto", *train_args],
         run / "train_summary.json")
    step(f"Collect examples for {name}", "collect_top", ["--sae", str(run), *mode], run / "examples" / "meta.json")
    step(f"Evaluate {name}", "eval", ["--sae", str(run), *mode], run / "eval.json")
    step(f"Labeling for {name}" + (" (offline stand-in)" if fake_labels else " (dry run, nothing is sent)"),
         "autointerp", ["--run", str(run), *(["--fake"] if fake_labels else [])], None)
    return run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", action="store_true", help="random GPT-2 weights and synthetic text: no downloads")
    ap.add_argument("--tiny", action="store_true", help="small budgets; with --random also a 4-layer, 64-dim model")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: cuda if available)")
    ap.add_argument("--out", type=Path, default=RUNS_DIR, help="folder that holds the runs (default: research/runs)")
    ap.add_argument("--name", default=None, help=f"run name (default: {FULL_NAME}, or pipeline-test with --tiny/--random)")
    ap.add_argument("--skip-smoke", action="store_true", help="do not run the smoke test before the real run")
    ap.add_argument("--training-tokens", type=int, default=None, help="override the training length (default 200M)")
    args = ap.parse_args(argv)

    mode = [flag for flag, on in (("--random", args.random), ("--tiny", args.tiny)) if on]
    if args.device:
        mode += ["--device", args.device]
    real = not (args.random or args.tiny)
    name = args.name or (FULL_NAME if real else "pipeline-test")
    train_args = ["--training-tokens", str(args.training_tokens)] if args.training_tokens else []
    started = time.time()

    if real and not args.skip_smoke:
        smoke_mode = ["--tiny", *(["--device", args.device] if args.device else [])]
        pipeline("smoke", args.out, smoke_mode, fake_labels=False, train_args=[])
        print("\nSmoke test passed: model, dataset, open SAE and GPU all work. Starting the real run.", flush=True)
    run = pipeline(name, args.out, mode, fake_labels=args.random, train_args=train_args)

    print(f"\nPhase 1 pipeline finished in {(time.time() - started) / 60:.1f} min. Results: {run}")
    print((run / "eval.md").read_text())
    if not args.random:
        print("Next, the only paid step left (started by you, needs ANTHROPIC_API_KEY):\n"
              f"  uv run python autointerp.py --run {run} --sample 200 --submit   # pilot, a few cents\n"
              f"  uv run python autointerp.py --run {run} --submit                # all concepts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
