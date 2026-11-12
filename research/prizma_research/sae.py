"""Sparse autoencoder (SAE) reference math and the on-disk format the site reads.

The site encodes the residual stream in JavaScript, so the format is deliberately dumb: raw
little-endian arrays plus a manifest that says how to use them. ``encode`` below is the reference
the browser implementation (``web/src/engine/sae.ts``) must reproduce.

Layout of ``artifacts/models/<name>/sae/`` (see ``artifacts/README.md``):

    manifest.json         SaeSettings, shapes, file names, version, self-check, verification
    encoder.f16.bin       [d_sae, d_model] fp16; row f is the encoder weight of feature f (W_enc[:, f])
    biases.f32.bin        b_enc [d_sae] followed by b_dec [d_model], fp32
    decoder/NNNN.f16.bin  rows [NNNN * shard, (NNNN + 1) * shard) of W_dec, [rows, d_model] fp16
    labels.json           optional: {"source": ..., "labels": [str | null] * d_sae}
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

FORMAT = 1
BOS_TOKEN_ID = 50256  # <|endoftext|>, which GPT-2 SAEs are trained with at position 0
SELFCHECK_TOP = 10


@dataclass
class SaeSettings:
    """How to turn a residual vector into feature activations."""

    activation: str  # "relu" (L1 SAEs) or "topk"
    k: int | None  # topk only: number of features kept per position
    center_resid: bool  # subtract the mean over d_model first (TransformerLens center_writing_weights)
    apply_b_dec_to_input: bool  # subtract b_dec before the encoder
    prepend_bos: bool  # the SAE was trained on contexts that start with <|endoftext|>


def settings_from_cfg(cfg: dict) -> SaeSettings:
    """Read the encode recipe from a SAELens ``cfg.json`` (old flat layout or v6 with ``metadata``).

    Missing keys get the SAELens defaults for legacy SAEs, which is what the open GPT-2 SAEs
    (``jbloom/GPT2-Small-SAEs-Reformatted``) need: ReLU, b_dec subtracted from the input, BOS
    prepended, and a model loaded with TransformerLens' ``center_writing_weights=True``.
    """
    meta = cfg.get("metadata") or {}

    def get(key: str, default):
        return cfg.get(key, meta.get(key, default))

    architecture = get("architecture", "standard")
    if architecture not in ("standard", "topk"):
        raise ValueError(f"unsupported SAE architecture {architecture!r} (supported: standard, topk)")
    if get("normalize_activations", "none") not in ("none", None, False):
        raise ValueError("normalize_activations must be 'none': fold the scaling into the weights before saving")
    if get("rescale_acts_by_decoder_norm", False):
        raise ValueError("rescale_acts_by_decoder_norm is set: call sae.fold_W_dec_norm() before saving")

    k = None
    if architecture == "topk":
        k = get("k", (get("activation_fn_kwargs", None) or {}).get("k"))
        if not k:
            raise ValueError("topk SAE without k in cfg.json")

    # TransformerLens centers the writing weights unless told otherwise, and old SAELens runs did
    # not record the kwargs at all, so only an explicit False means "raw residual".
    model_kwargs = get("model_from_pretrained_kwargs", None) or {}
    return SaeSettings(
        activation="topk" if architecture == "topk" else "relu",
        k=int(k) if k else None,
        center_resid=bool(model_kwargs.get("center_writing_weights", True)),
        apply_b_dec_to_input=bool(get("apply_b_dec_to_input", True)),
        prepend_bos=bool(get("prepend_bos", True)),
    )


def encode(x: torch.Tensor, w: dict[str, torch.Tensor], s: SaeSettings) -> torch.Tensor:
    """x [..., d_model] (residual as Hugging Face GPT-2 computes it) -> activations [..., d_sae]."""
    if s.center_resid:
        x = x - x.mean(dim=-1, keepdim=True)
    if s.apply_b_dec_to_input:
        x = x - w["b_dec"]
    pre = x @ w["W_enc"] + w["b_enc"]
    if s.activation == "topk":
        top = torch.topk(pre, s.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values.relu())
    return pre.relu()


def decode(acts: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruction of the SAE input (the centered residual when ``center_resid`` is set)."""
    return acts @ w["W_dec"] + w["b_dec"]


def explained_variance(x: torch.Tensor, w: dict[str, torch.Tensor], s: SaeSettings) -> tuple[float, float]:
    """(explained variance, mean L0) of the SAE on residuals x [N, d_model]."""
    acts = encode(x, w, s)
    target = x - x.mean(dim=-1, keepdim=True) if s.center_resid else x
    resid_var = (target - decode(acts, w)).pow(2).sum()
    total_var = (target - target.mean(dim=0)).pow(2).sum()
    return float(1 - resid_var / total_var), float((acts > 0).float().sum(-1).mean())


def _f16(t: torch.Tensor) -> np.ndarray:
    with np.errstate(over="ignore"):
        a = t.detach().cpu().numpy().astype("<f2")
    if not np.isfinite(a).all():
        raise ValueError("weights overflow fp16")
    return a


def write_weights(out_dir: Path, w: dict[str, torch.Tensor], shard_size: int = 256) -> dict:
    """Write the binary files; returns the manifest entries that describe them."""
    d_model, d_sae = w["W_enc"].shape
    assert w["W_dec"].shape == (d_sae, d_model) and w["b_enc"].shape == (d_sae,) and w["b_dec"].shape == (d_model,)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = _f16(w["W_enc"].T.contiguous())
    biases = torch.cat([w["b_enc"], w["b_dec"]]).detach().cpu().numpy().astype("<f4")
    (out_dir / "encoder.f16.bin").write_bytes(encoder.tobytes())
    (out_dir / "biases.f32.bin").write_bytes(biases.tobytes())

    dec_dir = out_dir / "decoder"
    shutil.rmtree(dec_dir, ignore_errors=True)  # a smaller SAE must not leave stale shards behind
    dec_dir.mkdir()
    decoder = _f16(w["W_dec"])
    for start in range(0, d_sae, shard_size):
        (dec_dir / f"{start // shard_size:04d}.f16.bin").write_bytes(decoder[start : start + shard_size].tobytes())

    digest = hashlib.sha256(encoder.tobytes() + biases.tobytes() + decoder.tobytes()).hexdigest()
    return {
        "d_model": d_model,
        "d_sae": d_sae,
        "version": digest[:12],
        "files": {"encoder": "encoder.f16.bin", "biases": "biases.f32.bin", "decoder": "decoder/{shard}.f16.bin"},
        "decoder_shard_size": shard_size,
        "bytes": {
            "encoder.f16.bin": encoder.nbytes,
            "biases.f32.bin": biases.nbytes,
            "decoder": decoder.nbytes,
        },
    }


def read_weights(sae_dir: Path, manifest: dict) -> dict[str, torch.Tensor]:
    """Load the weights exactly as the browser sees them (fp16-rounded, as fp32 tensors)."""
    d_model, d_sae = manifest["d_model"], manifest["d_sae"]
    files = manifest["files"]
    encoder = np.frombuffer((sae_dir / files["encoder"]).read_bytes(), dtype="<f2").reshape(d_sae, d_model)
    biases = np.frombuffer((sae_dir / files["biases"]).read_bytes(), dtype="<f4")
    n_shards = -(-d_sae // manifest["decoder_shard_size"])
    decoder = np.concatenate([
        np.frombuffer((sae_dir / files["decoder"].format(shard=f"{i:04d}")).read_bytes(), dtype="<f2")
        for i in range(n_shards)
    ]).reshape(d_sae, d_model)
    return {
        "W_enc": torch.from_numpy(encoder.astype(np.float32)).T.contiguous(),
        "b_enc": torch.from_numpy(biases[:d_sae].copy()),
        "b_dec": torch.from_numpy(biases[d_sae:].copy()),
        "W_dec": torch.from_numpy(decoder.astype(np.float32)),
    }


def selfcheck(x: torch.Tensor, w: dict[str, torch.Tensor], s: SaeSettings) -> dict:
    """One residual vector and its strongest features; the site re-computes this when it loads the SAE."""
    resid = [round(float(v), 5) for v in x]
    top = torch.topk(encode(torch.tensor(resid), w, s), SELFCHECK_TOP)
    return {"resid": resid, "features": top.indices.tolist(), "acts": [float(v) for v in top.values]}


def build_manifest(*, name: str, source: str, hook: str, settings: SaeSettings, files: dict,
                   neuronpedia: str | None = None) -> dict:
    manifest = {"format": FORMAT, "name": name, "source": source, "hook": hook, **asdict(settings)}
    if settings.k is None:
        del manifest["k"]
    if settings.prepend_bos:
        manifest["bos_token_id"] = BOS_TOKEN_ID
    if neuronpedia:
        manifest["neuronpedia"] = neuronpedia
    manifest.update(files)
    return manifest


def read_manifest(sae_dir: Path) -> tuple[dict, SaeSettings]:
    manifest = json.loads((sae_dir / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"unsupported SAE artifact format {manifest.get('format')!r}")
    settings = SaeSettings(
        activation=manifest["activation"],
        k=manifest.get("k"),
        center_resid=manifest["center_resid"],
        apply_b_dec_to_input=manifest["apply_b_dec_to_input"],
        prepend_bos=manifest["prepend_bos"],
    )
    return manifest, settings
