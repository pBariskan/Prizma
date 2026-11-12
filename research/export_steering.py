"""Build steering vectors for the split point and write them next to the ONNX parts.

Two sources of directions:

1. Activation addition ("ActAdd", Turner et al. 2023): the difference between the residual
   stream of two contrasting prompts (e.g. " Love" minus " Hate") at the split layer.
   Needs only GPT-2 itself.
2. Decoder directions of an open GPT-2 SAE (Joseph Bloom's residual-stream SAEs,
   ``jbloom/GPT2-Small-SAEs-Reformatted``, hook ``blocks.6.hook_resid_pre``). A curated set
   (``DEFAULT_SAE_FEATURES``) is included for GPT-2 small at layer 6; add more with
   ``--sae-feature ID[:label]``. Look ids up on Neuronpedia (model gpt2-small, SAE set res-jb, layer 6).

    uv run python export_steering.py
    uv run python export_steering.py --sae-feature 17946:"Musical instruments"
    uv run python export_steering.py --no-sae                 # ActAdd only, no SAE download
    uv run python export_steering.py --random --model-dir ../artifacts/models/gpt2-random-tiny  # tests

Output: ``<model-dir>/steering.json`` with unit-norm directions and, per vector, the strength
range that works.
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
NEURONPEDIA = "https://www.neuronpedia.org/gpt2-small/6-res-jb"

# Contrast pairs for ActAdd. The direction is the difference of the mean residual over each prompt,
# so the two prompts must differ in the concept itself: Turner et al.'s " I talk about weddings
# constantly" vs. " I do not talk about weddings" only works when added position by position, and
# as a mean difference it encodes "do not", not weddings. Topics come from the SAE features below.
DEFAULT_PAIRS = [
    ("love-hate", "Love vs. hate", " Love", " Hate"),
    ("happy-sad", "Happy vs. sad", " Happy", " Sad"),
]

# Features of the open layer-6 SAE whose Neuronpedia explanation matches what they do when added
# to the residual stream. Each was checked on real GPT-2 with three unrelated prompts
# (docs/phase0-verification.md): the topic shows up from about 0.25x, is clear at 0.5x, and the
# text collapses into repetition at 1x and above.
DEFAULT_SAE_FEATURES = [
    (12102, "Weddings"),
    (6330, "Christmas"),
    (16285, "Space missions"),
    (8888, "Cooking and recipes"),
    (16860, "God and faith"),
]

# Strengths are in units of the typical residual norm. Measured on real GPT-2: both kinds are
# clear at 0.5x; ActAdd vectors stay readable up to about 1x and fall apart from 1.5x, single SAE
# features are a sharper push and repeat one word from 1x.
STRENGTH = {
    "actadd": {"range": [-1.5, 1.5], "default_strength": 0.5},
    "sae": {"range": [-1.0, 1.0], "default_strength": 0.5},
}

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
    ap.add_argument("--sae-feature", action="append", default=[], metavar="ID[:LABEL]",
                    help="an extra feature of the open SAE, on top of the curated ones")
    ap.add_argument("--no-sae", action="store_true", help="ActAdd vectors only (no SAE download)")
    ap.add_argument("--random", action="store_true", help="random weights (must match the exported model)")
    args = ap.parse_args()

    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    layer = manifest["split_layer"]
    # The bundled SAE only fits GPT-2 small at layer 6.
    sae_fits = not args.random and manifest["d_model"] == 768 and layer == 6
    if args.sae_feature and not sae_fits:
        ap.error("the bundled SAE is for GPT-2 small at layer 6")
    sae_features = [] if args.no_sae or not sae_fits else [(fid, label) for fid, label in DEFAULT_SAE_FEATURES]
    for spec in args.sae_feature:
        fid, _, label = spec.partition(":")
        sae_features.append((int(fid), label or f"SAE feature {fid}"))

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

    # Same convention as the browser: every sequence starts with BOS when the manifest says so.
    bos = [manifest["bos_token_id"]] if manifest.get("prepend_bos") else []

    def ids_of(text: str) -> torch.Tensor:
        return torch.tensor([[*bos, *encode(text)]], dtype=torch.long)

    with torch.no_grad():
        # Typical residual norm at this layer, ignoring position 0 (BOS, or without it the first token:
        # GPT-2 uses that position as an attention sink and its norm is ~35x larger).
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
                **STRENGTH["actadd"],
                "vector": (diff / diff.norm()).tolist(),
            })

    if sae_features:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(SAE_REPO, f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors")
        w_dec = load_file(path)["W_dec"]  # [n_features, d_model]
        for fid, label in sae_features:
            d = w_dec[fid]
            vectors.append({
                "id": f"sae-{fid}",
                "label": label,
                "source": "sae",
                "detail": f"{SAE_REPO} blocks.{layer}.hook_resid_pre feature {fid}",
                "url": f"{NEURONPEDIA}/{fid}",
                "raw_norm": float(d.norm()),
                **STRENGTH["sae"],
                "vector": (d / d.norm()).tolist(),
            })

    out = {
        "hook": manifest["hook"],
        "d_model": manifest["d_model"],
        "typical_resid_norm": typical_norm,
        # Fallback slider range, in units of the typical residual norm, for vectors without their own.
        "suggested_range": [-2.0, 2.0],
        "vectors": vectors,
    }
    (args.model_dir / "steering.json").write_text(json.dumps(out))
    print(f"Wrote {len(vectors)} steering vectors to {args.model_dir / 'steering.json'} "
          f"(typical resid norm {typical_norm:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
