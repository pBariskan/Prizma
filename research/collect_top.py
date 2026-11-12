"""Collect the top activating examples of every SAE concept over ~10M held-out tokens.

    uv run python collect_top.py --sae runs/gpt2-l6-topk16384-k32     # needs a GPU and huggingface.co
    uv run python collect_top.py --sae runs/<name> --tiny             # 300k tokens, smoke test
    uv run python collect_top.py --sae runs/<name> --random --tiny    # offline CPU test

Contexts are 64 tokens (BOS + 63), the longest sequence the site ever runs. For every concept it
keeps the 20 strongest contexts, 10 random contexts where it fires, and firing statistics; see
``prizma_research/examples.py``. Output: ``<sae folder>/examples/``, read by ``autointerp.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from prizma_research.common import RUNS_DIR, add_model_args, fill_defaults, load_lm_from_args
from prizma_research.data import EXAMPLES_DATA_FILES, add_data_args, iter_contexts, iter_token_batches
from prizma_research.examples import ExampleCollector, Examples
from prizma_research.lm import EOT_ID, resid_at
from prizma_research.sae import load_sae

FULL = dict(n_tokens=10_000_000, context_size=64, batch_size=64, pool_size=1024)
TINY = dict(n_tokens=300_000, pool_size=256)
RANDOM_TINY = dict(n_tokens=60_000, context_size=32, pool_size=128)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    add_data_args(ap, default_files=EXAMPLES_DATA_FILES)
    ap.add_argument("--sae", required=True, help="SAELens folder from train_sae.py, 'jbloom', or 'repo:subfolder'")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <sae folder>/examples)")
    ap.add_argument("--n-tokens", type=int, default=None, help="tokens to scan (default 10M)")
    ap.add_argument("--context-size", type=int, default=None, help="tokens per context incl. BOS (default 64)")
    ap.add_argument("--batch-size", type=int, default=None, help="contexts per forward pass (default 64)")
    ap.add_argument("--top-k", type=int, default=20, help="strongest contexts kept per concept")
    ap.add_argument("--n-sample", type=int, default=10, help="random firing contexts kept per concept")
    ap.add_argument("--pool-size", type=int, default=None, help="contexts in the pool used for non-firing examples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    fill_defaults(args, FULL, TINY, RANDOM_TINY)

    lm = load_lm_from_args(args)
    device = lm.device
    sae = load_sae(args.sae, layer=lm.layer).to(device)
    if sae.d_in != lm.d_model:
        raise SystemExit(f"SAE input size {sae.d_in} does not match {lm.name} (d_model {lm.d_model})")
    out = args.out or (Path(args.sae) if Path(args.sae).is_dir() else RUNS_DIR / sae.name.replace("/", "--")) / "examples"

    collector = ExampleCollector(sae.d_sae, args.context_size, top_k=args.top_k, n_sample=args.n_sample,
                                 pool_size=args.pool_size, device=device, seed=args.seed)
    # --seed + 1: with --random the synthetic stream must differ from the one train_sae.py used.
    contexts = iter_contexts(lm, context_size=args.context_size, dataset=args.dataset, data_files=args.data_files,
                             seed=args.seed + 1)
    print(f"Collecting examples for {sae.name} ({sae.d_sae} concepts) on {lm.hook_name} of {lm.name}: "
          f"{args.n_tokens / 1e6:.2f}M tokens in contexts of {args.context_size} on {device}")
    started = time.time()
    seen = 0
    with torch.no_grad():
        for i, tokens in enumerate(iter_token_batches(contexts, batch_size=args.batch_size, n_tokens=args.n_tokens,
                                                      device=device)):
            acts = sae.encode(resid_at(lm.gpt, tokens, lm.layer))
            collector.add(tokens, acts, valid=tokens != EOT_ID)  # never BOS or a document boundary
            seen += tokens.numel()
            if i % 50 == 0:
                rate = seen / max(time.time() - started, 1e-9)
                print(f"[collect] {seen / 1e6:7.2f}M tokens  {rate / 1e3:.1f}k tok/s", flush=True)

    data = collector.result()
    ids = data["tokens"].unique().tolist()
    density = data["density"]
    meta = {
        "sae": sae.name,
        "model": lm.name,
        "hook": lm.hook_name,
        "d_sae": sae.d_sae,
        "context_size": args.context_size,
        "n_contexts": collector.n_contexts,
        "n_tokens": seen,
        "n_valid_tokens": collector.n_valid_tokens,
        "top_k": args.top_k,
        "n_sample": args.n_sample,
        "dataset": "synthetic" if lm.random else args.dataset,
        "data_files": None if lm.random else args.data_files,
        "never_fired": int((data["active_contexts"] == 0).sum()),
        "mean_l0": float(density.sum()),
        "seconds": time.time() - started,
    }
    Examples(data, {i: lm.tokenizer.decode([i]) for i in ids}, meta).save(out)
    print(f"Wrote {out} in {meta['seconds'] / 60:.1f} min: {collector.n_contexts} contexts, "
          f"mean L0 {meta['mean_l0']:.1f}, {meta['never_fired']} of {sae.d_sae} concepts never fired")
    return 0


if __name__ == "__main__":
    sys.exit(main())
