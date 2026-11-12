"""The streaming example collector must agree with a brute-force pass over all activations."""

import torch

from prizma_research.examples import ExampleCollector, Examples


def random_sparse_acts(n_ctx: int, T: int, d_sae: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    acts = torch.rand(n_ctx, T, d_sae, generator=g)
    acts = acts * (torch.rand(n_ctx, T, d_sae, generator=g) < 0.02)  # ~2% of entries fire
    acts[:, :, 0] = 0  # concept 0 never fires
    acts[:, :, 1] *= torch.arange(n_ctx)[:, None] < 3  # concept 1 fires in at most 3 contexts
    return acts


def collect(acts: torch.Tensor, tokens: torch.Tensor, batch: int, **kw) -> dict:
    collector = ExampleCollector(acts.shape[2], acts.shape[1], **kw)
    for i in range(0, len(acts), batch):
        collector.add(tokens[i:i + batch], acts[i:i + batch], valid=tokens[i:i + batch] != 50256)
    return collector.result()


def test_collector_matches_brute_force():
    n_ctx, T, d_sae, K = 90, 12, 40, 5
    acts = random_sparse_acts(n_ctx, T, d_sae, seed=0)
    tokens = torch.randint(0, 255, (n_ctx, T))
    tokens[:, 0] = 50256
    data = collect(acts, tokens, batch=16, top_k=K, n_sample=4, pool_size=32)

    valid_acts = acts.clone()
    valid_acts[:, 0] = 0  # BOS never counts
    peak = valid_acts.amax(1)  # [n_ctx, d_sae]
    for f in range(d_sae):
        want = peak[:, f].topk(K)
        n_firing = int((peak[:, f] > 0).sum())
        got_ctx = data["top_ctx"][f]
        assert int((got_ctx >= 0).sum()) == min(K, n_firing)
        for slot in range(min(K, n_firing)):
            c = int(got_ctx[slot])
            assert abs(float(peak[c, f]) - float(want.values[slot])) < 1e-3  # strongest first
            assert torch.allclose(data["top_acts"][f, slot].float(), valid_acts[c, :, f], atol=1e-3)
        # The random sample only ever holds contexts where the concept fires, without repeats.
        sample = [int(c) for c in data["sample_ctx"][f] if c >= 0]
        assert len(sample) == min(4, n_firing) == len(set(sample))
        assert all(peak[c, f] > 0 for c in sample)

    n_valid = n_ctx * (T - 1)
    assert torch.allclose(data["density"], (valid_acts > 0).sum((0, 1)) / n_valid, atol=1e-6)
    assert torch.allclose(data["max_act"], valid_acts.amax((0, 1)), atol=1e-6)
    assert torch.equal(data["active_contexts"], (peak > 0).sum(0))
    assert torch.allclose(data["pool_max_act"].float(), peak[:32].T, atol=1e-3)
    assert int(data["active_contexts"][0]) == 0 and int((data["top_ctx"][0] >= 0).sum()) == 0


def test_examples_windows_and_negatives(tmp_path):
    n_ctx, T, d_sae = 40, 12, 8
    acts = random_sparse_acts(n_ctx, T, d_sae, seed=1)
    tokens = torch.randint(65, 91, (n_ctx, T))
    tokens[:, 0] = 50256
    data = collect(acts, tokens, batch=8, top_k=3, n_sample=2, pool_size=16)
    vocab = {i: chr(i) for i in range(65, 91)} | {50256: "<|endoftext|>"}
    Examples(data, vocab, {"d_sae": d_sae}).save(tmp_path)
    ex = Examples.load(tmp_path)

    top = ex.top(2, before=3, after=1)
    assert 1 <= len(top) <= 3 and top[0].peak >= top[-1].peak > 0
    for e in top:
        assert len(e.tokens) == len(e.acts) <= 5 and "<|endoftext|>" not in e.tokens
        assert e.acts.index(e.peak) >= len(e.acts) - 2  # at most one token after the peak
    for e in ex.negatives(2, 5, seed=0, before=3, after=1):
        assert float(ex.data["pool_max_act"][2, e.context]) == 0 and set(e.acts) == {0.0}
    assert ex.top(0) == [] and ex.stats(0)["density"] == 0
