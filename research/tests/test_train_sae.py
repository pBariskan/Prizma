"""train_sae.py end to end on the offline test model, and the minimal SAE against SAELens."""

import json

import torch

from prizma_research.data import iter_contexts
from prizma_research.lm import EOT_ID, HookedGPT2, load_lm, resid_at
from prizma_research.sae import SAE, center


def test_tiny_run_writes_a_saelens_folder_that_learned_something(tiny_run):
    cfg = json.loads((tiny_run / "cfg.json").read_text())
    assert cfg["architecture"] == "topk" and cfg["k"] == 8
    assert cfg["normalize_activations"] == "none"  # the training-time scaling is folded into the weights
    assert cfg["metadata"]["prepend_bos"] is True
    assert cfg["metadata"]["model_from_pretrained_kwargs"] == {"center_writing_weights": False}
    assert cfg["metadata"]["hook_name"] == "blocks.2.hook_resid_pre"

    summary = json.loads((tiny_run / "train_summary.json").read_text())
    assert summary["check_after_reload"]["explained_variance"] > 0.5
    assert summary["check_after_reload"]["l0"] <= 8
    log = [json.loads(line) for line in (tiny_run / "train_log.jsonl").read_text().splitlines()]
    assert log[-1]["mse_loss"] < log[0]["mse_loss"]


@torch.no_grad()
def test_minimal_sae_matches_saelens(tiny_run):
    from sae_lens import SAE as SaeLensSAE

    ours = SAE.load(tiny_run)
    reference = SaeLensSAE.load_from_disk(str(tiny_run), device="cpu")
    lm = load_lm(random=True, tiny=True)
    tokens = torch.tensor([ctx for ctx, _ in zip(iter_contexts(lm, context_size=32), range(8))])
    x = resid_at(lm.gpt, tokens, lm.layer)

    acts = ours.encode(x)
    assert acts.shape == (8, 32, ours.d_sae)
    assert int((acts != 0).sum(-1).max()) <= 8
    assert torch.allclose(acts, reference.encode(x), atol=1e-5)
    assert torch.allclose(ours.decode(acts), reference.decode(reference.encode(x)), atol=1e-5)
    assert not ours.center_input and ours.prepend_bos


@torch.no_grad()
def test_hooked_wrapper_returns_the_part_a_residual():
    from prizma_research.gpt2_split import PartA

    lm = load_lm(random=True, tiny=True)
    tokens = torch.randint(0, 50257, (2, 12))
    hooked = HookedGPT2(lm.gpt, lm.layer)
    _, cache = hooked.run_with_cache(tokens, names_filter=[hooked.hook_name], stop_at_layer=lm.layer + 1,
                                     prepend_bos=False)
    part_a_resid, _ = PartA(lm.gpt, lm.layer)(lm.gpt.wte(tokens))
    assert torch.allclose(cache[hooked.hook_name], part_a_resid, atol=1e-6)


def test_contexts_start_with_bos_and_have_the_requested_length():
    lm = load_lm(random=True, tiny=True)
    contexts = [ctx for ctx, _ in zip(iter_contexts(lm, context_size=16), range(50))]
    assert all(len(ctx) == 16 and ctx[0] == EOT_ID for ctx in contexts)
    assert contexts[0] != contexts[1]


def test_centered_sae_removes_the_mean_before_encoding():
    sae = SAE.random_init(16, 64, center_input=True)
    x = torch.randn(4, 16)
    shifted = x + 3.0  # a shift along the all-ones direction is invisible to a centered SAE
    assert torch.allclose(sae.encode(x), sae.encode(shifted), atol=1e-5)
    assert torch.allclose(sae.reconstruct(shifted) - sae.reconstruct(x), torch.full((4, 16), 3.0), atol=1e-4)
    assert torch.allclose(center(shifted), center(x), atol=1e-5)
