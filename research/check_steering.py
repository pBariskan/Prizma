"""Reproduce the browser's steering pipeline in Python and print original vs. steered text.

Runs the exported ONNX parts exactly like the web worker does (BOS, fp16 embedding lookup, part A,
``resid += strength * typical_norm * direction`` at every position, part B, greedy decoding), so
the output should match the site token for token. Used to pick strength ranges and to write
``docs/phase0-verification.md``.

    uv run python check_steering.py                                  # every vector at its default strength
    uv run python check_steering.py --vector sae-12102 --strengths 0.25 0.5 0.75 1
    uv run python check_steering.py --markdown > /tmp/steering.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ["I think that you are", "The weather today is", "My favourite thing to do is"]


class SplitModel:
    def __init__(self, model_dir: Path):
        self.manifest = json.loads((model_dir / "manifest.json").read_text())
        self.steering = json.loads((model_dir / "steering.json").read_text())
        files = self.manifest["files"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.part_a = ort.InferenceSession(str(model_dir / files["part_a"]), opts)
        self.part_b = ort.InferenceSession(str(model_dir / files["part_b"]), opts)
        self.table = np.frombuffer((model_dir / files["embeddings"]).read_bytes(), dtype="<f2").reshape(
            -1, self.manifest["d_model"])
        self.vectors = {v["id"]: v for v in self.steering["vectors"]}

    def generate(self, ids: list[int], new_tokens: int, vector_id: str | None = None, strength: float = 0.0) -> list[int]:
        ids = list(ids)
        add = None
        if vector_id and strength:
            direction = np.asarray(self.vectors[vector_id]["vector"], dtype=np.float32)
            add = strength * self.steering["typical_resid_norm"] * direction
        for _ in range(new_tokens):
            embeds = self.table[np.asarray([ids])].astype(np.float32)
            (resid,) = self.part_a.run(["resid"], {"tok_embeds": embeds})
            if add is not None:
                resid = resid + add
            (last_logits,) = self.part_b.run(["last_logits"], {"resid": resid})
            ids.append(int(last_logits[0].argmax()))
        return ids[-new_tokens:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "models" / "gpt2-l6")
    ap.add_argument("--prompt", action="append", help="repeatable; default: three unrelated prompts")
    ap.add_argument("--vector", action="append", help="repeatable vector id; default: all")
    ap.add_argument("--strengths", type=float, nargs="+", help="default: each vector's default_strength")
    ap.add_argument("--new-tokens", type=int, default=20)
    ap.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = ap.parse_args()

    from transformers import GPT2TokenizerFast

    model = SplitModel(args.model_dir)
    tok = GPT2TokenizerFast.from_pretrained(model.manifest["source"])
    bos = [model.manifest["bos_token_id"]] if model.manifest.get("prepend_bos") else []
    prompts = args.prompt or DEFAULT_PROMPTS
    show = lambda ids: tok.decode(ids).replace("\n", "⏎")  # noqa: E731

    def row(prompt: str, label: str, strength: str, text: str) -> None:
        if args.markdown:
            print(f"| {prompt} | {label} | {strength} | {text.replace('|', '¦')} |")
        else:
            print(f"  [{strength:>6}] {label:28s} {prompt!r} -> {text!r}")

    if args.markdown:
        print("| Prompt | Vector | Strength | Continuation |\n| --- | --- | --- | --- |")
    originals = {p: model.generate([*bos, *tok.encode(p)], args.new_tokens) for p in prompts}
    for p in prompts:
        row(p, "(original)", "0", show(originals[p]))
    changed = total = 0
    for vid in args.vector or list(model.vectors):
        v = model.vectors[vid]
        for strength in args.strengths or [v.get("default_strength", 1.0)]:
            for p in prompts:
                out = model.generate([*bos, *tok.encode(p)], args.new_tokens, vid, strength)
                changed += out != originals[p]
                total += 1
                row(p, v["label"], f"{strength:+.2f}x", show(out))
    print(f"\nSteering changed the continuation in {changed} of {total} runs.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
