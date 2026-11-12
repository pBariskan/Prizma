"""A minimal sparse autoencoder in plain PyTorch: the math the browser runs.

``eval.py`` and ``collect_top.py`` use this class instead of SAELens' own modules on purpose. It
is the reference implementation of what the web engine does with the SAE artifact, and a test
checks it against SAELens on a freshly trained SAE.

    acts  = activation((x - b_dec) @ W_enc + b_enc)     # ReLU, or keep the k largest then ReLU
    x_hat = acts @ W_dec + b_dec

It reads the SAELens on-disk format (``cfg.json`` + ``sae_weights.safetensors``), both the
current one that ``train_sae.py`` writes and the older one of Joseph Bloom's open GPT-2 SAEs.

Input space. Our own SAE is trained on the raw residual stream that part A outputs. The open
``res-jb`` SAEs were trained on a TransformerLens model with ``center_writing_weights=True``,
whose residual stream is the raw one minus its mean over the 768 dimensions of every position.
``center_input`` records which of the two an SAE expects; ``encode`` applies it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

JBLOOM_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
WEIGHTS_FILE = "sae_weights.safetensors"
CFG_FILE = "cfg.json"


def center(x: torch.Tensor) -> torch.Tensor:
    """Remove the mean over the model dimension (what ``center_writing_weights`` does to the stream)."""
    return x - x.mean(dim=-1, keepdim=True)


@dataclass
class SAE:
    W_enc: torch.Tensor  # [d_in, d_sae]
    b_enc: torch.Tensor  # [d_sae]
    W_dec: torch.Tensor  # [d_sae, d_in]
    b_dec: torch.Tensor  # [d_in]
    architecture: str = "standard"  # "standard" (ReLU) or "topk"
    k: int | None = None
    apply_b_dec_to_input: bool = True
    center_input: bool = False
    prepend_bos: bool = True
    name: str = "sae"
    cfg: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.architecture not in ("standard", "topk"):
            raise NotImplementedError(f"unsupported SAE architecture {self.architecture!r}")
        if self.architecture == "topk" and not self.k:
            raise ValueError("a topk SAE needs k")
        d_in, d_sae = self.W_enc.shape
        if self.W_dec.shape != (d_sae, d_in) or self.b_enc.shape != (d_sae,) or self.b_dec.shape != (d_in,):
            raise ValueError("inconsistent SAE weight shapes")

    @property
    def d_in(self) -> int:
        return self.W_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    def to(self, device: torch.device | str) -> "SAE":
        for attr in ("W_enc", "b_enc", "W_dec", "b_dec"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def prepare(self, resid: torch.Tensor) -> torch.Tensor:
        """Raw residual stream -> the space this SAE was trained in."""
        return center(resid) if self.center_input else resid

    def encode(self, resid: torch.Tensor) -> torch.Tensor:
        """Raw residual [..., d_in] -> concept activations [..., d_sae] (dense, mostly zeros)."""
        x = self.prepare(resid.float())
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        if self.architecture == "standard":
            return pre.relu()
        values, indices = pre.topk(self.k, dim=-1, sorted=False)
        return torch.zeros_like(pre).scatter_(-1, indices, values.relu())

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Concept activations -> reconstruction in the SAE's input space (see ``prepare``)."""
        return acts @ self.W_dec + self.b_dec

    def reconstruct(self, resid: torch.Tensor) -> torch.Tensor:
        """Reconstruction that can replace ``resid`` in the model.

        For a ``center_input`` SAE the per-position mean is added back. The model would not notice
        either way (every later read goes through a LayerNorm, which removes that mean).
        """
        x_hat = self.decode(self.encode(resid))
        if self.center_input:
            x_hat = x_hat + resid.float().mean(dim=-1, keepdim=True)
        return x_hat

    # ---- loading -----------------------------------------------------------------------------

    @classmethod
    def from_weights(cls, weights: dict[str, torch.Tensor], cfg: dict, *, name: str,
                     center_input: bool | None = None) -> "SAE":
        """Build from SAELens-format weights and config (current or pre-6.0 layout)."""
        meta = cfg.get("metadata") or {}
        architecture = cfg.get("architecture")
        k = cfg.get("k") or (cfg.get("activation_fn_kwargs") or {}).get("k")
        if architecture is None:  # pre-6.0 configs name the activation function instead
            activation = cfg.get("activation_fn_str") or cfg.get("activation_fn") or "relu"
            architecture = "topk" if activation == "topk" else "standard"
        normalize = cfg.get("normalize_activations") or "none"  # old configs store a bool
        if normalize != "none":
            raise NotImplementedError(
                f"normalize_activations={normalize!r} is not folded into the weights; "
                "SAELens folds 'expected_average_only_in' at the end of training")
        pretrained_kwargs = cfg.get("model_from_pretrained_kwargs") or meta.get("model_from_pretrained_kwargs") or {}
        if center_input is None:
            center_input = bool(pretrained_kwargs.get("center_writing_weights", False))
        prepend_bos = cfg.get("prepend_bos", meta.get("prepend_bos", True))

        missing = {"W_enc", "b_enc", "W_dec", "b_dec"} - set(weights)
        if missing:
            raise ValueError(f"SAE weights are missing {sorted(missing)}")
        W_enc, b_enc, W_dec, b_dec = (weights[n].float().clone() for n in ("W_enc", "b_enc", "W_dec", "b_dec"))
        if cfg.get("rescale_acts_by_decoder_norm"):
            # SAELens TopK option: pre-activations are scaled by the decoder row norms. Fold the norms
            # into the weights so the runtime math above is all there is. (After training SAELens has
            # already done this and the norms are 1, which makes this a no-op.)
            norms = W_dec.norm(dim=-1).clamp(min=1e-8)
            W_enc, b_enc, W_dec = W_enc * norms, b_enc * norms, W_dec / norms[:, None]
        return cls(W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, architecture=architecture,
                   k=int(k) if k else None, apply_b_dec_to_input=bool(cfg.get("apply_b_dec_to_input", True)),
                   center_input=center_input, prepend_bos=bool(prepend_bos), name=name, cfg=cfg)

    @classmethod
    def load(cls, path: str | Path, *, center_input: bool | None = None) -> "SAE":
        """Load a local SAELens folder (``cfg.json`` + ``sae_weights.safetensors``)."""
        from safetensors.torch import load_file

        path = Path(path)
        cfg = json.loads((path / CFG_FILE).read_text())
        return cls.from_weights(load_file(str(path / WEIGHTS_FILE)), cfg, name=path.name, center_input=center_input)

    @classmethod
    def from_hub(cls, repo_id: str, subfolder: str, *, center_input: bool | None = None) -> "SAE":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg = json.loads(Path(hf_hub_download(repo_id, f"{subfolder}/{CFG_FILE}")).read_text())
        weights = load_file(hf_hub_download(repo_id, f"{subfolder}/{WEIGHTS_FILE}"))
        return cls.from_weights(weights, cfg, name=f"{repo_id}/{subfolder}", center_input=center_input)

    @classmethod
    def jbloom(cls, layer: int = 6) -> "SAE":
        """Joseph Bloom's open residual-stream SAE for GPT-2 small (``gpt2-small-res-jb``).

        Its ``cfg.json`` predates the fields that say so, but SAELens' registry records that these
        SAEs expect the centered residual stream and a BOS token (``pretrained_saes.yaml``:
        ``center_writing_weights: true``; ``prepend_bos`` defaults to true for old configs).
        """
        sae = cls.from_hub(JBLOOM_REPO, f"blocks.{layer}.hook_resid_pre", center_input=True)
        sae.name = f"jbloom-res-jb-l{layer}"
        return sae

    @classmethod
    def random_init(cls, d_in: int, d_sae: int, *, architecture: str = "standard", k: int | None = None,
                    seed: int = 0, center_input: bool = False, name: str = "random-sae") -> "SAE":
        """Untrained SAE with tied unit-norm weights; the offline stand-in for the open baseline."""
        g = torch.Generator().manual_seed(seed)
        W_dec = torch.randn(d_sae, d_in, generator=g)
        W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)
        return cls(W_enc=W_dec.T.clone(), b_enc=torch.zeros(d_sae), W_dec=W_dec, b_dec=torch.zeros(d_in),
                   architecture=architecture, k=k, center_input=center_input, name=name)


def load_sae(spec: str, *, layer: int) -> SAE:
    """``jbloom``, a local SAELens folder, or ``<hub repo>:<subfolder>``."""
    if spec == "jbloom":
        return SAE.jbloom(layer)
    if Path(spec).exists():
        return SAE.load(spec)
    if ":" in spec:
        repo_id, subfolder = spec.split(":", 1)
        return SAE.from_hub(repo_id, subfolder)
    raise FileNotFoundError(f"no SAE at {spec!r} (expected 'jbloom', a folder, or 'repo:subfolder')")
