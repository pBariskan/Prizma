"""Export a sparse autoencoder (SAE) for the split point into the format the site reads.

    uv run python export_sae.py                                   # open SAE: jbloom/GPT2-Small-SAEs-Reformatted
    uv run python export_sae.py --sae-path runs/prizma-topk-16k   # our own SAE (a SAELens save folder)
    uv run python export_sae.py --random --model-dir ../artifacts/models/gpt2-random-tiny   # tests, no download

Writes ``<model-dir>/sae/`` (format in ``artifacts/README.md`` and ``prizma_research/sae_artifact.py``).
Swapping the SAE the site uses means running this script on another SAE and uploading the folder;
no web code changes. ``--folder sae-own`` writes next to the current SAE instead of replacing it,
and ``/?sae=sae-own`` shows that one, so two SAEs can be compared on the same model.

The export verifies itself: it reloads the fp16 files it wrote, runs the model on a few sentences,
and writes explained variance and L0 into ``manifest.json``. It also checks the residual-centering
convention, because an SAE trained on a TransformerLens model sees a mean-centered residual stream
while the browser has the Hugging Face one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from prizma_research import sae_artifact as sae_format
from prizma_research.gpt2_split import GPT2
from prizma_research.sae_artifact import BOS_TOKEN_ID, SaeSettings

REPO_ROOT = Path(__file__).resolve().parent.parent
OPEN_SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"

CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969, astronauts landed on the Moon for the first time.",
    "My favourite recipe uses tomatoes, garlic and a little olive oil.",
    "The stock market fell sharply after the announcement on Tuesday.",
    "She opened the door and saw her brother standing in the rain.",
    "Scientists have discovered a new species of frog in the Amazon rainforest.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "The president said on Monday that the government would cut taxes next year.",
    "I love you so much, and I can't wait for our wedding in June.",
    "To install the package, run the following command in your terminal.",
    "The team won the championship after a dramatic overtime victory.",
    "Paris is the capital of France, and Berlin is the capital of Germany.",
]


def load_sae(args, manifest: dict) -> tuple[dict[str, torch.Tensor], dict, str, str | None]:
    """Returns (weights, SAELens cfg, human-readable source, Neuronpedia id)."""
    from safetensors.torch import load_file

    layer = manifest["split_layer"]
    if args.sae_path:
        cfg = json.loads((args.sae_path / "cfg.json").read_text())
        weights = load_file(args.sae_path / "sae_weights.safetensors")
        return weights, cfg, args.sae_path.name, None

    if args.random:
        d_model, d_sae = manifest["d_model"], args.d_sae
        g = torch.Generator().manual_seed(0)
        w_dec = torch.nn.functional.normalize(torch.randn(d_sae, d_model, generator=g), dim=-1)
        weights = {
            "W_enc": w_dec.T.clone(),
            "b_enc": -0.5 * torch.rand(d_sae, generator=g),
            "W_dec": w_dec,
            "b_dec": 0.1 * torch.randn(d_model, generator=g),
        }
        cfg = {"architecture": args.architecture, "k": args.k, "model_from_pretrained_kwargs": {"center_writing_weights": False}}
        return weights, cfg, "random", None

    from huggingface_hub import hf_hub_download

    sae_id = args.sae_id or f"blocks.{layer}.hook_resid_pre"
    cfg = json.loads(Path(hf_hub_download(args.sae_repo, f"{sae_id}/cfg.json")).read_text())
    weights = load_file(hf_hub_download(args.sae_repo, f"{sae_id}/sae_weights.safetensors"))
    neuronpedia = f"gpt2-small/{layer}-res-jb" if args.sae_repo == OPEN_SAE_REPO else None
    return weights, cfg, f"{args.sae_repo} {sae_id}", neuronpedia


def calibration_residuals(manifest: dict, random: bool, prepend_bos: bool) -> torch.Tensor:
    """Residuals at the split layer for the calibration texts, [N, d_model], without position 0.

    Position 0 is dropped either way: with BOS it is the BOS token, without it it is GPT-2's
    attention sink; neither looks like the positions the site shows concepts for.
    """
    from transformers import GPT2Config as HFConfig
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if random:
        # Rebuild the random model export_onnx.py made (same seed, same config).
        torch.manual_seed(0)
        tiny = manifest["d_model"] == 64
        hf = GPT2LMHeadModel(HFConfig(n_embd=64, n_layer=4, n_head=4, n_positions=128) if tiny else HFConfig()).eval()
        encode = lambda s: [ord(c) % manifest["vocab_size"] for c in s]  # noqa: E731 (no tokenizer download)
    else:
        hf = GPT2LMHeadModel.from_pretrained(manifest["source"]).eval()
        encode = GPT2TokenizerFast.from_pretrained(manifest["source"]).encode
    gpt = GPT2.from_hf(hf)

    rows = []
    with torch.no_grad():
        for text in CALIBRATION_TEXTS:
            ids = ([BOS_TOKEN_ID] if prepend_bos else []) + encode(text)
            x, _ = gpt.run_blocks(gpt.embed(torch.tensor([ids])), 0, manifest["split_layer"])
            rows.append(x[0, 1:])
    return torch.cat(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "models" / "gpt2-l6")
    ap.add_argument("--sae-repo", default=OPEN_SAE_REPO, help="Hugging Face repo with SAELens SAEs")
    ap.add_argument("--sae-id", default=None, help="folder inside the repo (default: blocks.<L>.hook_resid_pre)")
    ap.add_argument("--sae-path", type=Path, default=None, help="local SAELens folder (cfg.json + sae_weights.safetensors)")
    ap.add_argument("--folder", default="sae",
                    help="folder inside model-dir to write; the site reads 'sae', another one with ?sae=<folder>")
    ap.add_argument("--name", default=None, help="display name stored in the manifest")
    ap.add_argument("--neuronpedia", default=None, help="Neuronpedia id for feature links, e.g. gpt2-small/6-res-jb")
    ap.add_argument("--shard-size", type=int, default=256, help="decoder rows per shard file")
    ap.add_argument("--center-resid", action=argparse.BooleanOptionalAction, default=None,
                    help="override: the SAE expects the mean-centered residual (TransformerLens default)")
    ap.add_argument("--prepend-bos", action=argparse.BooleanOptionalAction, default=None,
                    help="override: the SAE expects contexts that start with <|endoftext|>")
    ap.add_argument("--random", action="store_true",
                    help="model-dir holds a --random model; without --sae-path the SAE is random too")
    ap.add_argument("--d-sae", type=int, default=512, help="with --random: number of features")
    ap.add_argument("--architecture", choices=["standard", "topk"], default="standard", help="with --random")
    ap.add_argument("--k", type=int, default=32, help="with --random --architecture topk")
    args = ap.parse_args()

    manifest = json.loads((args.model_dir / "manifest.json").read_text())
    weights, cfg, source, neuronpedia = load_sae(args, manifest)
    weights = {k: weights[k].float() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
    weights = sae_format.fold_decoder_norm(weights, cfg)
    if weights["W_enc"].shape[0] != manifest["d_model"]:
        ap.error(f"SAE d_in {weights['W_enc'].shape[0]} does not match the model's d_model {manifest['d_model']}")

    settings = sae_format.settings_from_cfg(cfg)
    if args.center_resid is not None:
        settings.center_resid = args.center_resid
    if args.prepend_bos is not None:
        settings.prepend_bos = args.prepend_bos

    out_dir = args.model_dir / args.folder
    files = sae_format.write_weights(out_dir, weights, args.shard_size)
    stored = sae_format.read_weights(out_dir, files)  # what the browser will actually compute with

    x = calibration_residuals(manifest, args.random, settings.prepend_bos)
    ev, l0 = sae_format.explained_variance(x, stored, settings)
    ev_fp32, _ = sae_format.explained_variance(x, weights, settings)
    flipped = SaeSettings(**{**settings.__dict__, "center_resid": not settings.center_resid})
    ev_flipped, _ = sae_format.explained_variance(x, stored, flipped)
    verification = {
        "positions": len(x),
        "explained_variance": ev,
        "explained_variance_fp32_weights": ev_fp32,
        "mean_l0": l0,
        # Same SAE with the other residual-centering convention. If this one is clearly higher,
        # center_resid is wrong for this SAE: re-run with --center-resid / --no-center-resid.
        "explained_variance_other_centering": ev_flipped,
    }
    print("verification:", json.dumps(verification))
    if not args.random and ev_flipped > ev + 0.02:
        print(f"WARNING: center_resid={settings.center_resid} explains less variance ({ev:.3f}) than the "
              f"opposite convention ({ev_flipped:.3f}). Check --center-resid.", file=sys.stderr)

    out = sae_format.build_manifest(
        name=args.name or source,
        source=source,
        hook=manifest["hook"],
        settings=settings,
        files=files,
        neuronpedia=args.neuronpedia or neuronpedia,
    )
    out["selfcheck"] = sae_format.selfcheck(x[len(x) // 2], stored, settings)
    out["verification"] = verification
    # Keep labels written earlier by export_labels.py / autointerp.py if they still fit this SAE.
    labels_path = out_dir / "labels.json"
    if labels_path.exists():
        if len(json.loads(labels_path.read_text())["labels"]) == files["d_sae"]:
            out["files"]["labels"] = "labels.json"
        else:
            print("WARNING: labels.json has a different number of features; ignoring it.", file=sys.stderr)
    (out_dir / "manifest.json").write_text(json.dumps(out, indent=2))
    total = sum(files["bytes"].values())
    print(f"Wrote {out_dir}: {files['d_sae']} features, {settings.activation}, "
          f"encoder {files['bytes']['encoder.f16.bin'] / 1e6:.1f} MB, total {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
