"""Export GPT-2 as two ONNX parts split at a residual-stream layer.

    uv run python export_onnx.py                      # real GPT-2 small, split at layer 6, 8-bit weights
    uv run python export_onnx.py --quantize none      # fp32 transformer blocks (~420 MB)
    uv run python export_onnx.py --random --tiny      # tiny random model, no download (for tests)

Writes to ``artifacts/models/<name>/``:
    embeddings.f16.bin  token-embedding table [vocab, d_model], raw little-endian fp16
    part_a.onnx     tok_embeds -> resid, attn
    part_b.onnx     resid -> last_logits, topk_probs, topk_ids, attn
                    (its unembedding matrix is read from embeddings.f16.bin as ONNX external data)
    manifest.json   shapes, split layer, file names, verification report; the contract the web engine reads
"""

from __future__ import annotations

import argparse
import hashlib
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
EMBEDDINGS_FILE = "embeddings.f16.bin"

# Real text for the self-check. Random token ids are a poor test: the next-token distribution after
# gibberish is nearly flat, so a tiny error flips the argmax and says little about real use.
VERIFY_PROMPTS = [
    "I think that you are",
    "The capital of France is",
    "Once upon a time, there was a little girl who",
    "The president of the United States said that",
    "In 1969, astronauts landed on the Moon for the first time, and",
    "She opened the door and saw",
    "Scientists have discovered a new species of",
    "The stock market fell sharply after",
    "To make a good cup of coffee, you need",
    "He looked at her and said, \"",
    "According to the report, the company",
    "When I was a child, my family used to",
    "The most important thing in a relationship is",
    "def fibonacci(n):\n    if n <",
    "The recipe calls for two cups of flour, one egg and",
    "On Saturday we went to the beach and",
]
GREEDY_PROMPTS = 8  # how many of the prompts also get a greedy continuation check
GREEDY_TOKENS = 12


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


def find_unembedding(model: onnx.ModelProto, cfg: GPT2Config) -> tuple[onnx.NodeProto, onnx.TensorProto]:
    """The MatMul that turns the final residual into logits, and its [d_model, vocab] weight."""
    weights = {i.name: i for i in model.graph.initializer if list(i.dims) == [cfg.n_embd, cfg.vocab_size]}
    nodes = [n for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in weights]
    if len(nodes) != 1:
        raise RuntimeError(f"expected exactly one unembedding MatMul, found {len(nodes)}")
    return nodes[0], weights[nodes[0].input[1]]


def quantize(path: Path, bits: int, cfg: GPT2Config) -> None:
    """Weight-only block quantization (MatMulNBits) of every MatMul with a constant weight.

    The unembedding matrix is left alone: it is by far the most sensitive weight (quantizing it to
    8 bits costs ~60x more KL than quantizing all 12 blocks), and ``tie_unembedding`` replaces it
    with the fp16 embedding table that the browser downloads anyway.
    """
    import logging

    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)

    model = onnx.load(str(path))
    skip = [find_unembedding(model, cfg)[0].name] if path.name == "part_b.onnx" else []
    # Asymmetric blocks cost one extra byte per 32 weights and cut the KL to Hugging Face by a third.
    q = MatMulNBitsQuantizer(model, bits=bits, block_size=32, is_symmetric=False, nodes_to_exclude=skip)
    q.process()
    q.model.save_model_to_file(str(path), use_external_data_format=False)


def tie_unembedding(path: Path, cfg: GPT2Config) -> None:
    """Make part B read its unembedding matrix from the fp16 embedding table (GPT-2 ties the two).

    The weight becomes an ONNX external-data initializer pointing at ``embeddings.f16.bin`` followed
    by Cast and Transpose, which ONNX Runtime folds into a constant when the session is created.
    The browser passes the bytes it already downloaded (``externalData`` session option), so the
    38.6M-parameter matrix is neither downloaded twice nor quantized.
    """
    from onnx import TensorProto, helper

    model = onnx.load(str(path))
    node, weight = find_unembedding(model, cfg)
    model.graph.initializer.remove(weight)

    table = TensorProto(name="wte_f16", data_type=TensorProto.FLOAT16, dims=[cfg.vocab_size, cfg.n_embd])
    table.data_location = TensorProto.EXTERNAL
    for key, value in {"location": EMBEDDINGS_FILE, "offset": 0, "length": cfg.vocab_size * cfg.n_embd * 2}.items():
        entry = table.external_data.add()
        entry.key, entry.value = key, str(value)
    model.graph.initializer.append(table)

    at = list(model.graph.node).index(node)
    model.graph.node.insert(at, helper.make_node("Cast", ["wte_f16"], ["wte_f32"], to=TensorProto.FLOAT, name="/unembed/Cast"))
    model.graph.node.insert(at + 1, helper.make_node("Transpose", ["wte_f32"], ["wte_t"], perm=[1, 0], name="/unembed/Transpose"))
    node.input[1] = "wte_t"
    onnx.save(model, str(path))


def verification_inputs(args, cfg: GPT2Config) -> list[list[int]]:
    """Token ids to check on: real sentences (with BOS) for real weights, seeded random ids otherwise."""
    if args.random:
        g = torch.Generator().manual_seed(0)
        return [torch.randint(0, cfg.vocab_size, (12,), generator=g).tolist() for _ in range(4)]
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(args.model)
    return [[tok.bos_token_id, *tok.encode(p)] for p in VERIFY_PROMPTS]


def verify(gpt: GPT2, hf, out_dir: Path, split: int, top_k: int, inputs: list[list[int]]) -> dict:
    """Check the split ONNX pipeline against the Hugging Face reference model.

    For every input: the last-position logits (max abs difference, KL divergence), and the top-1
    prediction at every position. For the first few inputs: whether greedy decoding produces the
    same continuation, which is what a visitor of the site would actually notice.
    """
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sa = ort.InferenceSession(str(out_dir / "part_a.onnx"), opts)
    sb = ort.InferenceSession(str(out_dir / "part_b.onnx"), opts)

    def onnx_run(ids: list[int]):
        embeds = lookup(out_dir / EMBEDDINGS_FILE, np.asarray([ids]), gpt.cfg.n_embd)
        resid, attn_a = sa.run(None, {"tok_embeds": embeds})
        last_logits, _, topk_ids, attn_b = sb.run(None, {"resid": resid})
        return last_logits[0], topk_ids[0], attn_a, attn_b

    def hf_logits(ids: list[int]) -> np.ndarray:
        with torch.no_grad():
            return hf(torch.tensor([ids])).logits[0].numpy()

    def greedy(next_id, ids: list[int]) -> list[int]:
        ids = list(ids)
        for _ in range(GREEDY_TOKENS):
            ids.append(int(next_id(ids)))
        return ids[-GREEDY_TOKENS:]

    def log_softmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64) - x.max()
        return x - np.log(np.exp(x).sum())

    torch_diff, onnx_diff, kls = [], [], []
    agree = positions = last_agree = 0
    for ids in inputs:
        ref = hf_logits(ids)
        with torch.no_grad():
            ours = gpt(torch.tensor([ids]))[0, -1].numpy()
        last_logits, topk_ids, attn_a, attn_b = onnx_run(ids)
        assert attn_a.shape[1] == split and attn_b.shape[1] == gpt.cfg.n_layer - split
        assert topk_ids.shape[-1] == top_k

        torch_diff.append(np.abs(ours - ref[-1]).max())
        onnx_diff.append(np.abs(last_logits - ref[-1]).max())
        agree += int((topk_ids[:, 0] == ref.argmax(-1)).sum())
        positions += len(ids)
        last_agree += int(last_logits.argmax() == ref[-1].argmax())
        p, q = log_softmax(ref[-1]), log_softmax(last_logits)
        kls.append(float((np.exp(p) * (p - q)).sum()))

    prefixes = []
    for ids in inputs[:GREEDY_PROMPTS]:
        want = greedy(lambda x: hf_logits(x)[-1].argmax(), ids)
        got = greedy(lambda x: onnx_run(x)[0].argmax(), ids)
        prefixes.append(next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), GREEDY_TOKENS))

    return {
        "prompts": len(inputs),
        "positions": positions,
        "torch_vs_hf_max_abs": float(max(torch_diff)),
        "onnx_vs_hf_max_abs": float(max(onnx_diff)),
        "kl_mean": float(np.mean(kls)),
        "kl_max": float(np.max(kls)),
        "top1_agreement": agree / positions,
        "argmax_match": last_agree == len(inputs),
        "greedy": {
            "prompts": len(prefixes),
            "tokens": GREEDY_TOKENS,
            "identical": sum(p == GREEDY_TOKENS for p in prefixes),
            "mean_matching_prefix": float(np.mean(prefixes)),
        },
        "attn_a_shape": list(attn_a.shape),
        "attn_b_shape": list(attn_b.shape),
        "topk_shape": [1, *topk_ids.shape],
    }


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
    ap.add_argument("--min-top1", type=float, default=0.97,
                    help="fail if the exported model agrees with Hugging Face on fewer top-1 predictions "
                         "(real weights only)")
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
    write_embeddings(gpt, out_dir / EMBEDDINGS_FILE)
    export_part(PartA(gpt, split), resid, out_dir / "part_a.onnx", "tok_embeds", ["resid", "attn"],
                {"tok_embeds": {1: "T"}, "resid": {1: "T"}, "attn": {3: "T", 4: "T"}})
    export_part(PartB(gpt, split, args.top_k), resid, out_dir / "part_b.onnx", "resid",
                ["last_logits", "topk_probs", "topk_ids", "attn"],
                {"resid": {1: "T"}, "topk_probs": {1: "T"}, "topk_ids": {1: "T"}, "attn": {3: "T", 4: "T"}})

    # Verify fp32 first: this isolates export bugs from quantization error.
    inputs = verification_inputs(args, cfg)
    report = {"fp32": verify(gpt, hf, out_dir, split, args.top_k, inputs)}
    print("fp32 check:", json.dumps(report["fp32"]))

    if args.quantize != "none":
        bits = 8 if args.quantize == "q8" else 4
        quantize(out_dir / "part_a.onnx", bits, cfg)
        quantize(out_dir / "part_b.onnx", bits, cfg)
    tie_unembedding(out_dir / "part_b.onnx", cfg)
    report["final"] = verify(gpt, hf, out_dir, split, args.top_k, inputs)
    print(f"{args.quantize} check:", json.dumps(report["final"]))

    files = [*sorted(out_dir.glob("*.onnx")), out_dir / EMBEDDINGS_FILE]
    sizes = {p.name: p.stat().st_size for p in files}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
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
        # Every sequence starts with <|endoftext|>: GPT-2 saw it between documents, the open SAEs
        # for this hook were trained that way, and otherwise the first real token is the attention sink.
        "prepend_bos": True,
        "bos_token_id": hf.config.bos_token_id,
        "top_k": args.top_k,
        "quantization": args.quantize,
        "files": {"embeddings": EMBEDDINGS_FILE, "part_a": "part_a.onnx", "part_b": "part_b.onnx"},
        # Files a part reads as ONNX external data; the engine passes their bytes when it creates the session.
        "external_data": {"part_b": [EMBEDDINGS_FILE]},
        "bytes": sizes,
        "sha256": hashes,  # lets the browser tell a cached file from an older export
        "verification": report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir} ({sum(sizes.values()) / 1e6:.1f} MB)")

    top1 = report["final"]["top1_agreement"]
    if not args.random and top1 < args.min_top1:
        print(f"ERROR: top-1 agreement with Hugging Face is {top1:.3f}, below --min-top1 {args.min_top1}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
