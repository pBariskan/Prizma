"""Per-concept example collection: the data behind labels, scores and the concept pages.

``ExampleCollector`` consumes dense SAE activations batch by batch and keeps, for every concept:

    top      the K contexts where it fires most strongly (one entry per context, at its peak)
    sample   R contexts drawn uniformly from all contexts where it fires at all (reservoir
             sampling with random keys); these are the held-out positives for scoring a label
    stats    density (share of tokens it fires on), mean and max activation, active contexts

plus a small pool of contexts with every concept's peak activation on them, from which the
scoring pass draws examples where a concept is known not to fire.

Everything is stored as tensors in ``examples.safetensors`` next to ``meta.json`` and
``vocab.json`` (token id -> display string), so ``autointerp.py`` needs neither the model nor the
tokenizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

EXAMPLES_FILE = "examples.safetensors"
META_FILE = "meta.json"
VOCAB_FILE = "vocab.json"


class ExampleCollector:
    def __init__(self, d_sae: int, context_size: int, *, top_k: int = 20, n_sample: int = 10, pool_size: int = 1024,
                 device: torch.device | str = "cpu", seed: int = 0):
        self.d_sae, self.context_size = d_sae, context_size
        self.top_k, self.n_sample, self.pool_size = top_k, n_sample, pool_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        def table(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (torch.zeros(d_sae, n, device=self.device),  # sort key: peak activation or random key
                    torch.full((d_sae, n), -1, dtype=torch.int32, device=self.device),  # context index
                    torch.zeros(d_sae, n, context_size, dtype=torch.float16, device=self.device))  # per-token acts

        self.top_key, self.top_ctx, self.top_acts = table(top_k)
        self.smp_key, self.smp_ctx, self.smp_acts = table(n_sample)
        self.pool_max = torch.zeros(d_sae, pool_size, dtype=torch.float16, device=self.device)
        self.fire_count = torch.zeros(d_sae, dtype=torch.float64, device=self.device)
        self.act_sum = torch.zeros(d_sae, dtype=torch.float64, device=self.device)
        self.max_act = torch.zeros(d_sae, device=self.device)
        self.active_contexts = torch.zeros(d_sae, dtype=torch.int64, device=self.device)
        self.n_valid_tokens = 0
        self.n_contexts = 0
        self.tokens: list[torch.Tensor] = []

    @torch.no_grad()
    def add(self, tokens: torch.Tensor, acts: torch.Tensor, valid: torch.Tensor) -> None:
        """tokens [B, T], acts [B, T, d_sae] (dense), valid [B, T] bool: positions that count."""
        B = tokens.shape[0]
        acts = acts * valid[..., None]
        self.n_valid_tokens += int(valid.sum())
        self.fire_count += (acts > 0).sum((0, 1))
        self.act_sum += acts.sum((0, 1), dtype=torch.float64)
        self.max_act = torch.maximum(self.max_act, acts.amax((0, 1)))

        peak = acts.amax(dim=1).T.contiguous()  # [d_sae, B]: each concept's peak on each context
        fires = peak > 0
        self.active_contexts += fires.sum(1)
        ctx_ids = (self.n_contexts + torch.arange(B, dtype=torch.int32, device=self.device)).expand(self.d_sae, B)
        cand_acts = acts.permute(2, 0, 1).to(torch.float16)  # [d_sae, B, T]

        self.top_key, self.top_ctx, self.top_acts = self._merge(
            self.top_key, self.top_ctx, self.top_acts, peak, ctx_ids, cand_acts)
        random_key = torch.rand(peak.shape, generator=self.generator, device=self.device).clamp(min=1e-9) * fires
        self.smp_key, self.smp_ctx, self.smp_acts = self._merge(
            self.smp_key, self.smp_ctx, self.smp_acts, random_key, ctx_ids, cand_acts)

        n_pool = min(B, self.pool_size - self.n_contexts)
        if n_pool > 0:
            self.pool_max[:, self.n_contexts:self.n_contexts + n_pool] = peak[:, :n_pool].to(torch.float16)
        self.n_contexts += B
        self.tokens.append(tokens.to(torch.int32).cpu())

    @staticmethod
    def _merge(key, ctx, acts, cand_key, cand_ctx, cand_acts):
        """Keep the n largest keys among the stored entries and the new batch (n = stored width)."""
        n, T = key.shape[1], acts.shape[2]
        all_key = torch.cat([key, cand_key], dim=1)
        new_key, sel = all_key.topk(n, dim=1)
        new_ctx = torch.cat([ctx, cand_ctx], dim=1).gather(1, sel)
        new_acts = torch.cat([acts, cand_acts], dim=1).gather(1, sel[..., None].expand(-1, -1, T))
        return new_key, new_ctx, new_acts

    def result(self) -> dict[str, torch.Tensor]:
        """Tensors for ``examples.safetensors``. Empty slots have context index -1."""
        top_ctx = torch.where(self.top_key > 0, self.top_ctx, torch.full_like(self.top_ctx, -1))
        smp_ctx = torch.where(self.smp_key > 0, self.smp_ctx, torch.full_like(self.smp_ctx, -1))
        fire = self.fire_count.clamp(min=1)
        out = {
            "tokens": torch.cat(self.tokens) if self.tokens else torch.zeros(0, self.context_size, dtype=torch.int32),
            "top_ctx": top_ctx,
            "top_acts": self.top_acts,
            "sample_ctx": smp_ctx,
            "sample_acts": self.smp_acts,
            "pool_max_act": self.pool_max[:, :min(self.pool_size, self.n_contexts)],
            "density": (self.fire_count / max(self.n_valid_tokens, 1)).float(),
            "mean_act": (self.act_sum / fire).float(),
            "max_act": self.max_act,
            "active_contexts": self.active_contexts,
        }
        return {name: t.cpu().contiguous() for name, t in out.items()}


@dataclass
class Example:
    """One context for one concept, cut to a window around the strongest token."""

    tokens: list[str]
    acts: list[float]
    context: int  # index of the context in the collection (lets callers drop duplicates)

    @property
    def peak(self) -> float:
        return max(self.acts, default=0.0)


@dataclass
class Examples:
    """What ``collect_top.py`` wrote, with the accessors ``autointerp.py`` needs."""

    data: dict[str, torch.Tensor]
    vocab: dict[int, str]
    meta: dict

    @classmethod
    def load(cls, folder: str | Path) -> "Examples":
        from safetensors.torch import load_file

        folder = Path(folder)
        vocab = {int(k): v for k, v in json.loads((folder / VOCAB_FILE).read_text()).items()}
        return cls(load_file(str(folder / EXAMPLES_FILE)), vocab, json.loads((folder / META_FILE).read_text()))

    def save(self, folder: str | Path) -> None:
        from safetensors.torch import save_file

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        save_file(self.data, str(folder / EXAMPLES_FILE))
        (folder / VOCAB_FILE).write_text(json.dumps({str(k): v for k, v in self.vocab.items()}, ensure_ascii=False))
        (folder / META_FILE).write_text(json.dumps(self.meta, indent=2) + "\n")

    @property
    def d_sae(self) -> int:
        return self.data["top_ctx"].shape[0]

    def _window(self, ctx: int, acts: torch.Tensor | None, peak: int, before: int, after: int) -> Example:
        tokens = self.data["tokens"][ctx]
        start, end = max(1, peak - before), min(len(tokens), peak + after + 1)  # position 0 is BOS
        values = [0.0] * (end - start) if acts is None else [round(float(a), 3) for a in acts[start:end]]
        return Example([self.vocab.get(int(t), "�") for t in tokens[start:end]], values, ctx)

    def _stored(self, ctx_key: str, acts_key: str, feature: int, before: int, after: int) -> list[Example]:
        ctxs, acts = self.data[ctx_key][feature], self.data[acts_key][feature].float()
        return [self._window(int(c), acts[i], int(acts[i].argmax()), before, after)
                for i, c in enumerate(ctxs) if c >= 0]

    def top(self, feature: int, *, before: int = 24, after: int = 8) -> list[Example]:
        """Strongest contexts first."""
        return self._stored("top_ctx", "top_acts", feature, before, after)

    def sample(self, feature: int, *, before: int = 24, after: int = 8) -> list[Example]:
        """Random contexts where the concept fires (may overlap with ``top`` for rare concepts)."""
        return self._stored("sample_ctx", "sample_acts", feature, before, after)

    def negatives(self, feature: int, n: int, *, seed: int, before: int = 24, after: int = 8) -> list[Example]:
        """Pool contexts on which the concept does not fire at all.

        Each is cut around a random position, so that it has the same shape as a firing example
        (which is cut around its peak) and length alone does not give it away.
        """
        silent = (self.data["pool_max_act"][feature] <= 0).nonzero().flatten()
        g = torch.Generator().manual_seed(seed * 1_000_003 + feature)
        picks = silent[torch.randperm(len(silent), generator=g)[:n]]
        T = self.data["tokens"].shape[1]
        centers = torch.randint(1, T, (len(picks),), generator=g) if T > 1 else torch.zeros(len(picks), dtype=torch.int64)
        return [self._window(int(c), None, int(p), before, after) for c, p in zip(picks, centers)]

    def stats(self, feature: int) -> dict:
        return {
            "density": float(self.data["density"][feature]),
            "max_act": float(self.data["max_act"][feature]),
            "mean_act": float(self.data["mean_act"][feature]),
            "active_contexts": int(self.data["active_contexts"][feature]),
        }
