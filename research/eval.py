"""Measure an SAE on held-out text and compare it with the open GPT-2 SAE.

    uv run python eval.py --sae runs/gpt2-l6-topk16384-k32      # vs Joseph Bloom's res-jb SAE; needs huggingface.co
    uv run python eval.py --sae runs/<name> --baseline none     # our SAE alone
    uv run python eval.py --sae runs/<name> --tiny              # 200k tokens, smoke test
    uv run python eval.py --sae runs/<name> --random --tiny     # offline CPU test; the baseline is an untrained SAE

Reports explained variance, L0, the language-model loss with the SAE spliced into the residual
stream, and the share of dead and dense concepts, on ~4M held-out tokens in 64-token contexts (the
shape the site runs). Definitions: ``prizma_research/metrics.py``. Output: ``eval.json`` and
``eval.md`` in the SAE folder, with the Phase 1 gate check against the baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from prizma_research.common import RUNS_DIR, add_model_args, fill_defaults, load_lm_from_args, write_json
from prizma_research.data import EVAL_DATA_FILES, add_data_args, iter_contexts, iter_token_batches
from prizma_research.lm import EOT_ID
from prizma_research.metrics import (
    GATE_EXPLAINED_VARIANCE_GAP,
    GATE_LOSS_RECOVERED_GAP,
    GATE_MAX_DEAD_FRACTION,
    Evaluator,
    gate,
    markdown_table,
)
from prizma_research.sae import SAE, load_sae

FULL = dict(n_tokens=4_000_000, context_size=64, batch_size=32)
TINY = dict(n_tokens=200_000)
RANDOM_TINY = dict(n_tokens=20_000, context_size=32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    add_data_args(ap, default_files=EVAL_DATA_FILES)
    ap.add_argument("--sae", required=True, help="SAELens folder from train_sae.py, 'jbloom', or 'repo:subfolder'")
    ap.add_argument("--baseline", default="jbloom",
                    help="SAE to compare with: 'jbloom' (default), a folder, 'repo:subfolder', or 'none'")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: the SAE folder)")
    ap.add_argument("--n-tokens", type=int, default=None, help="tokens to evaluate on (default 4M)")
    ap.add_argument("--context-size", type=int, default=None, help="tokens per context incl. BOS (default 64)")
    ap.add_argument("--batch-size", type=int, default=None, help="contexts per forward pass (default 32)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    fill_defaults(args, FULL, TINY, RANDOM_TINY)

    lm = load_lm_from_args(args)
    device = lm.device
    sae = load_sae(args.sae, layer=lm.layer).to(device)
    saes = {sae.name: sae}
    if args.baseline != "none":
        if lm.random:
            # No open SAE exists for random weights. An untrained ReLU SAE on the centered residual
            # stands in, so the offline run covers the same code paths as the real comparison.
            baseline = SAE.random_init(lm.d_model, 2 * lm.d_model, center_input=True, name="untrained-sae")
        else:
            baseline = load_sae(args.baseline, layer=lm.layer)
        saes[baseline.name] = baseline.to(device)
    for s in saes.values():
        if s.d_in != lm.d_model:
            raise SystemExit(f"SAE {s.name} has input size {s.d_in}, but {lm.name} has d_model {lm.d_model}")
    out = args.out or (Path(args.sae) if Path(args.sae).is_dir() else RUNS_DIR / sae.name.replace("/", "--"))

    # --seed + 2: with --random the synthetic stream must differ from training and example collection.
    contexts = iter_contexts(lm, context_size=args.context_size, dataset=args.dataset, data_files=args.data_files,
                             seed=args.seed + 2)
    print(f"Evaluating {', '.join(saes)} on {lm.hook_name} of {lm.name}: {args.n_tokens / 1e6:.2f}M tokens "
          f"in contexts of {args.context_size} on {device}")
    evaluator = Evaluator(lm, saes)
    started = time.time()
    for i, tokens in enumerate(iter_token_batches(contexts, batch_size=args.batch_size, n_tokens=args.n_tokens,
                                                  device=device)):
        evaluator.add(tokens, valid=tokens != EOT_ID)
        if i % 100 == 0:
            rate = evaluator.n_tokens / max(time.time() - started, 1e-9)
            print(f"[eval] {evaluator.n_tokens / 1e6:7.2f}M tokens  {rate / 1e3:.1f}k tok/s", flush=True)

    result = {
        "model": lm.name,
        "hook": lm.hook_name,
        "context_size": args.context_size,
        "dataset": "synthetic" if lm.random else args.dataset,
        "data_files": None if lm.random else args.data_files,
        "sae": sae.name,
        "baseline": None if len(saes) == 1 else list(saes)[1],
        **evaluator.result(),
        "seconds": time.time() - started,
    }
    table = markdown_table(result)
    report = [f"# SAE evaluation: {sae.name}", "",
              f"{result['n_valid_tokens']:,} held-out tokens of {result['dataset']} in contexts of "
              f"{args.context_size}, {lm.hook_name} of {lm.name}. Clean LM loss {result['clean_loss']:.3f} nats, "
              f"with the residual zeroed {result['zero_ablation_loss']:.3f}.", "", table, ""]
    if result["baseline"]:
        checks = gate(result["saes"][sae.name], result["saes"][result["baseline"]])
        result["gate"] = {"passed": all(checks.values()), "checks": checks}
        report += [f"Phase 1 gate (explained variance at most {GATE_EXPLAINED_VARIANCE_GAP} and loss recovered at most "
                   f"{100 * GATE_LOSS_RECOVERED_GAP:.0f} points below the baseline, at most "
                   f"{100 * GATE_MAX_DEAD_FRACTION:.0f}% dead concepts): "
                   + ("PASSED" if result["gate"]["passed"] else "NOT PASSED") + " ("
                   + ", ".join(f"{name} {'ok' if ok else 'FAILS'}" for name, ok in checks.items()) + ")", ""]
        if lm.random:
            report += ["The baseline of this offline run is an untrained SAE, so the gate only exercises the code.", ""]
        elif args.tiny:
            report += ["This is a --tiny run: a small SAE trained for minutes. It is not expected to pass the gate; "
                       "it only shows that every step works.", ""]
    write_json(out / "eval.json", result)
    (out / "eval.md").write_text("\n".join(report))
    print("\n" + "\n".join(report[2:]))
    print(f"Wrote {out / 'eval.json'} and {out / 'eval.md'} in {result['seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
