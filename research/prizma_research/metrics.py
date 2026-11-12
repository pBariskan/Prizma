"""SAE quality metrics on held-out text, accumulated batch by batch.

For every SAE (ours and the baseline) ``eval.py`` reports:

    explained variance   1 - |x - x_hat|^2 / |x - mean(x)|^2 over all tokens, in two spaces:
                         the raw residual stream (what the browser holds) and the residual with its
                         per-position mean removed (all the model ever reads, because every later read
                         goes through a LayerNorm; also the space the open SAE was trained in)
    L0                   concepts active per token
    LM loss              next-token loss with the residual replaced by the SAE's reconstruction,
                         next to the clean loss and the loss with the residual zeroed out;
                         loss recovered = (zero - spliced) / (zero - clean)
    dead / dense         share of concepts that never fire / fire on more than 1% of tokens

Positions holding ``<|endoftext|>`` (BOS and document boundaries) are left out everywhere: their
residual is an outlier that no SAE here is meant to reconstruct, and it is passed through untouched
when the model is run on a reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from prizma_research.lm import LM, logits_from_resid, resid_at
from prizma_research.sae import SAE, center

DENSE_THRESHOLD = 0.01  # a concept that fires on more than 1% of tokens is "dense"

# How we read the Phase 1 gate "metrics close to the open SAE": ours may be at most this much
# worse than the baseline (absolute), and at most this share of its concepts may be dead.
GATE_EXPLAINED_VARIANCE_GAP = 0.05
GATE_LOSS_RECOVERED_GAP = 0.02
GATE_MAX_DEAD_FRACTION = 0.10


def loss_sum(logits: torch.Tensor, tokens: torch.Tensor, positions: torch.Tensor) -> float:
    """Summed next-token loss (nats) of the predictions made at ``positions`` [B, T-1] (bool)."""
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1),
                          reduction="none")
    return float((nll * positions.reshape(-1)).sum(dtype=torch.float64))


class SpaceVariance:
    """Total variance of a stream of vectors: sum over dimensions of the variance of that dimension."""

    def __init__(self, d: int, device: torch.device) -> None:
        self.sum = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_sq = 0.0
        self.n = 0

    def add(self, x: torch.Tensor) -> None:  # [N, D]
        self.sum += x.sum(0, dtype=torch.float64)
        self.sum_sq += float(x.pow(2).sum(dtype=torch.float64))
        self.n += x.shape[0]

    @property
    def total(self) -> float:
        """Sum of squared distances to the mean vector."""
        return self.sum_sq - float(self.sum.pow(2).sum()) / max(self.n, 1)


class SAEMeter:
    """Running sums for one SAE."""

    def __init__(self, sae: SAE) -> None:
        self.sae = sae
        self.sq_err = self.sq_err_centered = self.l0_sum = self.loss = 0.0
        self.fire_count = torch.zeros(sae.d_sae, dtype=torch.float64, device=sae.device)

    @torch.no_grad()
    def add(self, lm: LM, tokens: torch.Tensor, resid: torch.Tensor, valid: torch.Tensor) -> None:
        acts = self.sae.encode(resid)
        x_hat = self.sae.decode(acts)
        if self.sae.center_input:  # back to the raw residual space (see SAE.reconstruct)
            x_hat = x_hat + resid.mean(dim=-1, keepdim=True)
        err = (resid - x_hat)[valid]
        self.sq_err += float(err.pow(2).sum(dtype=torch.float64))
        self.sq_err_centered += float(center(err).pow(2).sum(dtype=torch.float64))
        fired = (acts > 0)[valid]
        self.l0_sum += float(fired.sum())
        self.fire_count += fired.sum(0)
        spliced = torch.where(valid[..., None], x_hat, resid)
        self.loss += loss_sum(logits_from_resid(lm.gpt, spliced, lm.layer), tokens, valid[:, :-1])


class Evaluator:
    """Feed token batches with ``add``; ``result`` returns the metrics of every SAE."""

    def __init__(self, lm: LM, saes: dict[str, SAE]) -> None:
        self.lm = lm
        self.meters = {name: SAEMeter(sae) for name, sae in saes.items()}
        self.raw = SpaceVariance(lm.d_model, lm.device)
        self.centered = SpaceVariance(lm.d_model, lm.device)
        self.clean_loss = self.zero_loss = 0.0
        self.n_tokens = self.n_loss = 0

    @torch.no_grad()
    def add(self, tokens: torch.Tensor, valid: torch.Tensor) -> None:
        """tokens [B, T]; valid [B, T] bool: the positions that count (not <|endoftext|>)."""
        lm = self.lm
        resid = resid_at(lm.gpt, tokens, lm.layer)
        x = resid[valid]
        self.raw.add(x)
        self.centered.add(center(x))
        self.n_tokens += tokens.numel()
        self.n_loss += int(valid[:, :-1].sum())
        self.clean_loss += loss_sum(logits_from_resid(lm.gpt, resid, lm.layer), tokens, valid[:, :-1])
        zeroed = torch.where(valid[..., None], torch.zeros_like(resid), resid)
        self.zero_loss += loss_sum(logits_from_resid(lm.gpt, zeroed, lm.layer), tokens, valid[:, :-1])
        for meter in self.meters.values():
            meter.add(lm, tokens, resid, valid)

    def result(self) -> dict:
        n, n_loss = max(self.raw.n, 1), max(self.n_loss, 1)
        clean, zero = self.clean_loss / n_loss, self.zero_loss / n_loss
        saes = {}
        for name, m in self.meters.items():
            density = m.fire_count / n
            spliced = m.loss / n_loss
            saes[name] = {
                "name": m.sae.name,
                "d_sae": m.sae.d_sae,
                "architecture": m.sae.architecture,
                "k": m.sae.k,
                "center_input": m.sae.center_input,
                "explained_variance": 1 - m.sq_err / self.raw.total,
                "explained_variance_centered": 1 - m.sq_err_centered / self.centered.total,
                "mse": m.sq_err / n,
                "l0": m.l0_sum / n,
                "spliced_loss": spliced,
                "loss_increase": spliced - clean,
                # None when zeroing the residual costs nothing (cannot happen with a trained model).
                "loss_recovered": (zero - spliced) / (zero - clean) if zero - clean > 1e-9 else None,
                "dead_fraction": float((m.fire_count == 0).double().mean()),
                "dense_fraction": float((density > DENSE_THRESHOLD).double().mean()),
                "max_density": float(density.max()),
            }
        return {"n_tokens": self.n_tokens, "n_valid_tokens": self.raw.n, "clean_loss": clean,
                "zero_ablation_loss": zero, "saes": saes}


def gate(ours: dict, baseline: dict) -> dict[str, bool]:
    """Our reading of "metrics close to the open SAE" (thresholds at the top of this module)."""
    return {
        "explained_variance": ours["explained_variance_centered"]
                              >= baseline["explained_variance_centered"] - GATE_EXPLAINED_VARIANCE_GAP,
        "loss_recovered": None not in (ours["loss_recovered"], baseline["loss_recovered"])
                          and ours["loss_recovered"] >= baseline["loss_recovered"] - GATE_LOSS_RECOVERED_GAP,
        "dead_concepts": ours["dead_fraction"] <= GATE_MAX_DEAD_FRACTION,
    }


def markdown_table(result: dict) -> str:
    """The metrics of all SAEs side by side."""
    saes = result["saes"]
    rows = [
        ("Concepts", lambda s: f"{s['d_sae']:,}"),
        ("Activation", lambda s: f"TopK, k = {s['k']}" if s["architecture"] == "topk" else "ReLU"),
        ("Explained variance (raw residual)", lambda s: f"{s['explained_variance']:.3f}"),
        ("Explained variance (mean removed)", lambda s: f"{s['explained_variance_centered']:.3f}"),
        ("L0 (active concepts per token)", lambda s: f"{s['l0']:.1f}"),
        ("LM loss with the SAE spliced in", lambda s: f"{s['spliced_loss']:.3f}"),
        ("LM loss increase (nats)", lambda s: f"{s['loss_increase']:+.3f}"),
        ("LM loss recovered", lambda s: "n/a" if s["loss_recovered"] is None else f"{100 * s['loss_recovered']:.1f}%"),
        ("Dead concepts (never fire)", lambda s: f"{100 * s['dead_fraction']:.1f}%"),
        (f"Dense concepts (> {100 * DENSE_THRESHOLD:.0f}% of tokens)", lambda s: f"{100 * s['dense_fraction']:.1f}%"),
    ]
    lines = ["| Metric | " + " | ".join(saes) + " |", "| --- |" + " ---: |" * len(saes)]
    lines += [f"| {title} | " + " | ".join(cell(s) for s in saes.values()) + " |" for title, cell in rows]
    return "\n".join(lines)
