"""The split forward pass must reproduce the Hugging Face GPT-2 exactly."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from prizma_research.gpt2_split import GPT2, PartA, PartB


def tiny_models():
    torch.manual_seed(0)
    hf = GPT2LMHeadModel(GPT2Config(n_embd=64, n_layer=4, n_head=4, n_positions=64)).eval()
    return hf, GPT2.from_hf(hf)


@torch.no_grad()
def test_full_forward_matches_hf():
    hf, gpt = tiny_models()
    ids = torch.randint(0, 50257, (1, 10))
    assert torch.allclose(gpt(ids), hf(ids).logits, atol=1e-4)


@torch.no_grad()
def test_split_equals_full_and_steering_changes_output():
    hf, gpt = tiny_models()
    ids = torch.randint(0, 50257, (1, 10))
    resid, attn_a = PartA(gpt, 2)(gpt.wte(ids))
    last, topk_p, topk_i, attn_b = PartB(gpt, 2)(resid)
    assert torch.allclose(last, hf(ids).logits[:, -1], atol=1e-4)
    assert attn_a.shape == (1, 2, 4, 10, 10) and attn_b.shape == (1, 2, 4, 10, 10)
    # Attention rows are causal probability distributions.
    assert torch.allclose(attn_a.sum(-1), torch.ones(1, 2, 4, 10), atol=1e-5)
    assert attn_a[..., 0, 1:].abs().max() == 0
    # Adding a direction to the residual stream changes the prediction.
    steered, *_ = PartB(gpt, 2)(resid + 5.0 * torch.randn(64))
    assert not torch.allclose(steered, last)
