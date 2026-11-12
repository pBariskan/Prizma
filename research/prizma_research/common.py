"""Command-line plumbing shared by the Phase 1 scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prizma_research.lm import LM, load_lm

RESEARCH_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = RESEARCH_ROOT.parent
# Training runs (weights, checkpoints, logs, examples, labels) live here. Not committed, and not
# under artifacts/ either: artifacts/ holds only what the site downloads.
RUNS_DIR = RESEARCH_ROOT / "runs"


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default="gpt2", help="Hugging Face model id (default: gpt2)")
    ap.add_argument("--layer", type=int, default=None,
                    help="read blocks.{layer}.hook_resid_pre (default: 6, or 2 for the tiny random model)")
    ap.add_argument("--random", action="store_true",
                    help="random GPT-2 weights and synthetic text: no downloads, for tests")
    ap.add_argument("--tiny", action="store_true",
                    help="small budgets for a quick end-to-end check; with --random also a 4-layer, 64-dim model")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: cuda if available)")


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lm_from_args(args: argparse.Namespace) -> LM:
    return load_lm(args.model, random=args.random, tiny=args.random and args.tiny, layer=args.layer,
                   device=pick_device(args.device))


def fill_defaults(args: argparse.Namespace, full: dict, tiny: dict, random_tiny: dict) -> None:
    """Fill options left at ``None`` from the profile that matches --tiny / --random."""
    profile = dict(full)
    if args.tiny:
        profile.update(tiny)
        if args.random:
            profile.update(random_tiny)
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
