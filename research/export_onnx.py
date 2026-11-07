"""Export GPT-2 as two ONNX parts split at a residual-stream layer.

    uv run python export_onnx.py                      # real GPT-2 small, split at layer 6, 8-bit weights
    uv run python export_onnx.py --quantize none      # full fp32 weights (~650 MB)
    uv run python export_onnx.py --random --tiny      # tiny random model, no download (for tests)

Writes to ``artifacts/models/<name>/``:
    embeddings.f16.bin  token-embedding table [vocab, d_model], raw little-endian fp16
    part_a.onnx     tok_embeds -> resid, attn
    part_b.onnx     resid -> last_logits, topk_probs, topk_ids, attn
    manifest.json   shapes, split layer, file names; the contract the web engine reads
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from prizma_research.gpt2_split import GPT2, GPT2Config, PartA, PartB

REPO_ROOT = Path(__file__).resolve().parent.parent
OPSET = 17


def load_model(args) -> tuple[GPT2, object | None]:
    from transformers import GPT2Config as HFConfig
    from transformers import GPT2LMHeadModel

    if args.random:
        torch.manual_seed(0)
        if args.tiny:
            hf_cfg = HFConfig(n_embd=64, n_layer=4, n_head=4, n_positions=128)
        else:
            hf_cfg = HFConfig()
        hf = GPT2LMHeadModel(hf_cfg).eval()
    else:
        hf = GPT2LMHeadModel.from_pretrained(args.model).eval()
    return GPT2.from_hf(hf), hf


def export_part(module: torch.nn.Module, example: torch.Tensor, path: Path, input_name: str,
                output_names: list[str], dynamic_axes: dict) -> None:
    torch.onnx.export(
        module,
        (example,),
        str(path),
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


def write_embeddings(gpt: GPT2, path: Path) -> None:
    """Token-embedding table as raw little-endian fp16, shape [vocab, d_model]."""
    path.write_bytes(gpt.wte.weight.detach().numpy().astype("<f2").tobytes())


def lookup(path: Path, ids: np.ndarray, d_model: int) -> np.ndarray:
    """What the browser does: read rows of the fp16 table and widen them to fp32."""
    table = np.frombuffer(path.read_bytes(), dtype="<f2").reshape(-1, d_model)
    return table[ids].astype(np.float32)


def quantize(path: Path, bits: int) -> None:
    """Weight-only block quantization of every MatMul with a constant weight (MatMulNBits)."""
    import logging

    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)

    model = onnx.load(str(path))
    q = MatMulNBitsQuantizer(model, bits=bits, block_size=32, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(str(path), use_external_data_format=False)


def verify(gpt: GPT2, hf, out_dir: Path, split: int, top_k: int) -> dict:
    """Check the split ONNX pipeline against the Hugging Face reference model."""
    ids = torch.randint(0, gpt.cfg.vocab_size, (1, 12))
    with torch.no_grad():
        ref_logits = hf(ids).logits[0, -1].numpy()
        ours_logits = gpt(ids)[0, -1].numpy()

    sa = ort.InferenceSession(str(out_dir / "part_a.onnx"))
    sb = ort.InferenceSession(str(out_dir / "part_b.onnx"))
    embeds = lookup(out_dir / "embeddings.f16.bin", ids.numpy(), gpt.cfg.n_embd)
    resid, attn_a = sa.run(None, {"tok_embeds": embeds})
    last_logits, topk_probs, topk_ids, attn_b = sb.run(None, {"resid": resid})

    report = {
        "torch_vs_hf_max_abs": float(np.abs(ours_logits - ref_logits).max()),
        "onnx_vs_hf_max_abs": float(np.abs(last_logits[0] - ref_logits).max()),
        "argmax_match": bool(last_logits[0].argmax() == ref_logits.argmax()),
        "attn_a_shape": list(attn_a.shape),
        "attn_b_shape": list(attn_b.shape),
        "topk_shape": list(topk_ids.shape),
    }
    assert attn_a.shape[1] == split and attn_b.shape[1] == gpt.cfg.n_layer - split
    assert topk_ids.shape[-1] == top_k
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2", help="Hugging Face model id (default: gpt2)")
    ap.add_argument("--split-layer", type=int, default=6, help="residual stream entering this block")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--quantize", choices=["none", "q8", "q4"], default="q8")
    ap.add_argument("--random", action="store_true", help="random weights, no download")
    ap.add_argument("--tiny", action="store_true", help="with --random: a 4-layer, 64-dim model")
    ap.add_argument("--name", default=None, help="output folder name under artifacts/models/")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "models")
    args = ap.parse_args()

    gpt, hf = load_model(args)
    cfg: GPT2Config = gpt.cfg
    split = args.split_layer
    if not 0 < split < cfg.n_layer:
        ap.error(f"--split-layer must be in [1, {cfg.n_layer - 1}]")

    name = args.name or ("gpt2-random-tiny" if args.tiny else "gpt2-random" if args.random else
                         f"{args.model.replace('/', '--')}-l{split}")
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    T = 8
    resid = torch.zeros(1, T, cfg.n_embd)
    print(f"Exporting {name}: split at layer {split} of {cfg.n_layer}")
    write_embeddings(gpt, out_dir / "embeddings.f16.bin")
    export_part(PartA(gpt, split), resid, out_dir / "part_a.onnx", "tok_embeds", ["resid", "attn"],
                {"tok_embeds": {1: "T"}, "resid": {1: "T"}, "attn": {3: "T", 4: "T"}})
    export_part(PartB(gpt, split, args.top_k), resid, out_dir / "part_b.onnx", "resid",
                ["last_logits", "topk_probs", "topk_ids", "attn"],
                {"resid": {1: "T"}, "topk_probs": {1: "T"}, "topk_ids": {1: "T"}, "attn": {3: "T", 4: "T"}})

    # Verify fp32 first: this isolates export bugs from quantization error.
    report = {"fp32": verify(gpt, hf, out_dir, split, args.top_k)}
    print("fp32 check:", json.dumps(report["fp32"]))

    if args.quantize != "none":
        bits = 8 if args.quantize == "q8" else 4
        quantize(out_dir / "part_a.onnx", bits)
        quantize(out_dir / "part_b.onnx", bits)
    report["final"] = verify(gpt, hf, out_dir, split, args.top_k)
    print(f"{args.quantize} check:", json.dumps(report["final"]))

    sizes = {p.name: p.stat().st_size for p in [*out_dir.glob("*.onnx"), out_dir / "embeddings.f16.bin"]}
    manifest = {
        "name": name,
        "source": "random" if args.random else args.model,
        "split_layer": split,
        "hook": f"blocks.{split}.hook_resid_pre",
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "d_model": cfg.n_embd,
        "vocab_size": cfg.vocab_size,
        "n_positions": cfg.n_positions,
        "top_k": args.top_k,
        "quantization": args.quantize,
        "files": {"embeddings": "embeddings.f16.bin", "part_a": "part_a.onnx", "part_b": "part_b.onnx"},
        "bytes": sizes,
        "verification": report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir} ({sum(sizes.values()) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
