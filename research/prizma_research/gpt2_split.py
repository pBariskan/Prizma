"""A minimal, explicit GPT-2 forward pass split into two halves at a chosen layer.

Part A: token embeddings -> residual stream entering block `split_layer` (``blocks.{L}.hook_resid_pre``)
        plus the attention patterns of blocks [0, L).
Part B: residual stream -> next-token logits for the last position, top-k predictions for
        every position, plus the attention patterns of blocks [L, n_layer).

The browser runs A, edits the residual stream (SAE encoding, steering), then runs B.

We re-implement the forward pass in plain PyTorch instead of calling the Hugging Face
modules so that (1) the ONNX graph is small and predictable, (2) the causal mask is
explicit, and (3) the attention probabilities are first-class outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    layer_norm_epsilon: float = 1e-5

    @classmethod
    def from_hf(cls, hf_config) -> "GPT2Config":
        return cls(
            vocab_size=hf_config.vocab_size,
            n_positions=hf_config.n_positions,
            n_embd=hf_config.n_embd,
            n_layer=hf_config.n_layer,
            n_head=hf_config.n_head,
            layer_norm_epsilon=hf_config.layer_norm_epsilon,
        )


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """The tanh GELU approximation used by GPT-2."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        d = cfg.n_embd
        self.n_head = cfg.n_head
        self.eps = cfg.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(d, eps=self.eps)
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)
        self.ln_2 = nn.LayerNorm(d, eps=self.eps)
        self.c_fc = nn.Linear(d, 4 * d)
        self.mlp_proj = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, D] -> (x_out [B, T, D], attn_probs [B, H, T, T])."""
        B, T, D = x.shape
        H = self.n_head
        hd = D // H

        q, k, v = self.c_attn(self.ln_1(x)).split(D, dim=-1)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)

        attn = (probs @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.c_proj(attn)
        x = x + self.mlp_proj(gelu_new(self.c_fc(self.ln_2(x))))
        return x, probs


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)

    @classmethod
    def from_hf(cls, hf_model) -> "GPT2":
        """Copy weights from a transformers ``GPT2LMHeadModel``.

        Hugging Face GPT-2 uses ``Conv1D`` layers whose weights are stored as [in, out],
        so they are transposed into ``nn.Linear`` layout [out, in].
        """
        cfg = GPT2Config.from_hf(hf_model.config)
        model = cls(cfg)
        sd = hf_model.state_dict()
        p = "transformer."
        with torch.no_grad():
            model.wte.weight.copy_(sd[p + "wte.weight"])
            model.wpe.weight.copy_(sd[p + "wpe.weight"])
            model.ln_f.weight.copy_(sd[p + "ln_f.weight"])
            model.ln_f.bias.copy_(sd[p + "ln_f.bias"])
            for i, blk in enumerate(model.blocks):
                h = f"{p}h.{i}."
                blk.ln_1.weight.copy_(sd[h + "ln_1.weight"])
                blk.ln_1.bias.copy_(sd[h + "ln_1.bias"])
                blk.ln_2.weight.copy_(sd[h + "ln_2.weight"])
                blk.ln_2.bias.copy_(sd[h + "ln_2.bias"])
                blk.c_attn.weight.copy_(sd[h + "attn.c_attn.weight"].T)
                blk.c_attn.bias.copy_(sd[h + "attn.c_attn.bias"])
                blk.c_proj.weight.copy_(sd[h + "attn.c_proj.weight"].T)
                blk.c_proj.bias.copy_(sd[h + "attn.c_proj.bias"])
                blk.c_fc.weight.copy_(sd[h + "mlp.c_fc.weight"].T)
                blk.c_fc.bias.copy_(sd[h + "mlp.c_fc.bias"])
                blk.mlp_proj.weight.copy_(sd[h + "mlp.c_proj.weight"].T)
                blk.mlp_proj.bias.copy_(sd[h + "mlp.c_proj.bias"])
        return model.eval()

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.add_positions(self.wte(input_ids))

    def add_positions(self, tok_embeds: torch.Tensor) -> torch.Tensor:
        T = tok_embeds.shape[1]
        pos = torch.arange(T, device=tok_embeds.device).unsqueeze(0)
        return tok_embeds + self.wpe(pos)

    def run_blocks(self, x: torch.Tensor, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run blocks [start, end). Returns (x, attn [B, end-start, H, T, T])."""
        patterns = []
        for blk in self.blocks[start:end]:
            x, probs = blk(x)
            patterns.append(probs)
        return x, torch.stack(patterns, dim=1)

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln_f(x) @ self.wte.weight.T

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x, _ = self.run_blocks(self.embed(input_ids), 0, self.cfg.n_layer)
        return self.unembed(x)


class PartA(nn.Module):
    """tok_embeds [1, T, D] -> resid [1, T, D], attn [1, L, H, T, T].

    The token-embedding lookup itself happens outside the graph (the browser reads rows from an
    fp16 table), because fp16 Gather is not supported on every WebGPU device and an fp32 table
    would add ~150 MB to the download.
    """

    def __init__(self, gpt: GPT2, split_layer: int):
        super().__init__()
        self.gpt = gpt
        self.split_layer = split_layer

    def forward(self, tok_embeds: torch.Tensor):
        return self.gpt.run_blocks(self.gpt.add_positions(tok_embeds), 0, self.split_layer)


class PartB(nn.Module):
    """resid [1, T, D] -> last_logits [1, V], topk_probs/topk_ids [1, T, k], attn [1, n-L, H, T, T]."""

    def __init__(self, gpt: GPT2, split_layer: int, top_k: int = 10):
        super().__init__()
        self.gpt = gpt
        self.split_layer = split_layer
        self.top_k = top_k

    def forward(self, resid: torch.Tensor):
        x, attn = self.gpt.run_blocks(resid, self.split_layer, self.gpt.cfg.n_layer)
        logits = self.gpt.unembed(x)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, self.top_k, dim=-1)
        return logits[:, -1, :], topk_probs, topk_ids.to(torch.int32), attn
