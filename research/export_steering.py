"""Build steering vectors for the split point and write them next to the ONNX parts.

Two sources of directions are supported:

1. Activation addition ("ActAdd", Turner et al. 2023): the difference between the residual
   stream of two contrasting prompts (e.g. " Love" minus " Hate") at the split layer.
   Needs only GPT-2 itself, so it is the default for the Phase 0 prototype.
2. Decoder directions of an open GPT-2 SAE (Joseph Bloom's residual-stream SAEs,
   ``jbloom/GPT2-Small-SAEs-Reformatted``, hook ``blocks.6.hook_resid_pre``).
   Pass feature ids with ``--sae-feature ID[:label]``; look ids up on Neuronpedia
   (model gpt2-small, SAE set res-jb, layer 6).

    uv run python export_steering.py
    uv run python export_steering.py --sae-feature 1234:"weddings"
    uv run python export_steering.py --random --model-dir ../artifacts/models/gpt2-random-tiny  # tests

Output: ``<model-dir>/steering.json`` with unit-norm directions and a suggested slider range.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from prizma_research.gpt2_split import GPT2

REPO_ROOT = Path(__file__).resolve().parent.parent
SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"

DEFAULT_PAIRS = [
    ("love-hate", "Love vs. hate", " Love", " Hate"),
    ("weddings", "Talking about weddings", " I talk about weddings constantly", " I do not talk about weddings"),
    ("happy-sad", "Happy vs. sad", " Happy", " Sad"),
]

CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, astronauts landed on the Moon for the first time.",
    "My favourite recipe uses tomatoes, garlic and a little olive oil.",
    "The stock market fell sharply after the announcement on Tuesday.",
]


def resid_at(gpt: GPT2, ids: torch.Tensor, layer: int) -> torch.Tensor:
    x, _ = gpt.run_blocks(gpt.embed(ids), 0, layer)
    return x[0]  # [T, D]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "models" / "gpt2-l6")
    ap.add_argument("--sae-feature", action="append", default=[], metavar="ID[:LABEL]")
    ap.add_argument("--random", action="store_true", help="random weights (must match the exported model)")
    args = ap.parse_args()

    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    layer = manifest["split_layer"]

    from transformers import GPT2Config as HFConfig
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if args.random:
        # Rebuild the same random model export_onnx.py made (same seed, same config).
        torch.manual_seed(0)
        tiny = manifest["d_model"] == 64
        hf_cfg = HFConfig(n_embd=64, n_layer=4, n_head=4, n_positions=128) if tiny else HFConfig()
        hf = GPT2LMHeadModel(hf_cfg).eval()
        encode = lambda s: [ord(c) % manifest["vocab_size"] for c in s]  # noqa: E731 (no tokenizer download)
    else:
        hf = GPT2LMHeadModel.from_pretrained(manifest["source"]).eval()
        tok = GPT2TokenizerFast.from_pretrained(manifest["source"])
        encode = tok.encode
    gpt = GPT2.from_hf(hf)

    def ids_of(text: str) -> torch.Tensor:
        return torch.tensor([encode(text)], dtype=torch.long)

    with torch.no_grad():
        # Typical residual norm at this layer, ignoring position 0 (GPT-2's attention sink has a huge norm).
        norms = torch.cat([resid_at(gpt, ids_of(t), layer)[1:].norm(dim=-1) for t in CALIBRATION_TEXTS])
        typical_norm = float(norms.median())

        vectors = []
        for vid, label, pos, neg in DEFAULT_PAIRS:
            a, b = resid_at(gpt, ids_of(pos), layer), resid_at(gpt, ids_of(neg), layer)
            # Compare the mean over non-sink positions so prompts of different lengths work.
            diff = a[1:].mean(0) - b[1:].mean(0) if min(len(a), len(b)) > 1 else a[-1] - b[-1]
            vectors.append({
                "id": vid,
                "label": label,
                "source": "actadd",
                "detail": f"resid({pos!r}) - resid({neg!r}) at blocks.{layer}.hook_resid_pre",
                "raw_norm": float(diff.norm()),
                "vector": (diff / diff.norm()).tolist(),
            })

    if args.sae_feature:
        if manifest["d_model"] != 768 or layer != 6:
            ap.error("the bundled SAE is for GPT-2 small at layer 6")
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(SAE_REPO, f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors")
        w_dec = load_file(path)["W_dec"]  # [n_features, d_model]
        for spec in args.sae_feature:
            fid, _, label = spec.partition(":")
            d = w_dec[int(fid)]
            vectors.append({
                "id": f"sae-{fid}",
                "label": label or f"SAE feature {fid}",
                "source": "sae",
                "detail": f"{SAE_REPO} blocks.{layer}.hook_resid_pre feature {fid}",
                "raw_norm": float(d.norm()),
                "vector": (d / d.norm()).tolist(),
            })

    out = {
        "hook": manifest["hook"],
        "d_model": manifest["d_model"],
        "typical_resid_norm": typical_norm,
        # Slider is in units of the typical residual norm; beyond ~2x GPT-2 tends to fall apart.
        "suggested_range": [-2.0, 2.0],
        "vectors": vectors,
    }
    (args.model_dir / "steering.json").write_text(json.dumps(out))
    print(f"Wrote {len(vectors)} steering vectors to {args.model_dir / 'steering.json'} "
          f"(typical resid norm {typical_norm:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
