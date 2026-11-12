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


def test_characters_split_across_tokens_are_put_back_together(tmp_path):
    """The offline tokenizer has one token per byte, so every non-ASCII character is split, as in GPT-2."""
    from prizma_research.examples import merge_utf8, vocab_entry
    from prizma_research.lm import ByteTokenizer

    tokenizer = ByteTokenizer()
    text = "don’t 🙂 é"
    ids = tokenizer.encode(text)
    entries = [vocab_entry(tokenizer.token_bytes(i)) for i in ids]
    acts = [0.0] * len(ids)
    acts[3] = 2.0  # "’" is bytes 3, 4, 5: the concept fires on its first and last byte
    acts[5] = 5.0
    tokens, merged = merge_utf8(entries, acts)
    assert tokens == ["d", "o", "n", "’", "t", " ", "🙂", " ", "é"] and "".join(tokens) == text
    assert merged == [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # the largest activation of its parts
    # A character cut by the edge of the window is dropped, unless the concept fires on what is left of it.
    assert merge_utf8(entries[4:8], [0.0] * 4) == (["t", " "], [0.0, 0.0])
    assert merge_utf8(entries[2:4], [0.0] * 2) == (["n"], [0.0])
    assert merge_utf8(entries[2:4], [0.0, 3.0]) == (["n", "�"], [0.0, 3.0])
    assert merge_utf8(entries[5:7], [3.0, 0.0]) == (["�", "t"], [3.0, 0.0])

    # The vocabulary survives a save and load although single bytes are not valid UTF-8.
    n_ctx, T = 6, len(ids) + 1
    tokens_tensor = torch.tensor([[50256, *ids]] * n_ctx)
    acts_tensor = torch.zeros(n_ctx, T, 2)
    acts_tensor[:, 6, 1] = torch.arange(1, n_ctx + 1).float()  # concept 1 fires on the last byte of "’"
    data = collect(acts_tensor, tokens_tensor, batch=3, top_k=3, n_sample=2, pool_size=4)
    vocab = {i: vocab_entry(tokenizer.token_bytes(i)) for i in set(ids)} | {50256: "<|endoftext|>"}
    Examples(data, vocab, {}).save(tmp_path)
    best = Examples.load(tmp_path).top(1, before=24, after=20)[0]
    assert "".join(best.tokens) == text and best.tokens[3] == "’" and best.acts[3] == best.peak == 6.0
    assert "".join(Examples.load(tmp_path).top(1, before=24, after=8)[0].tokens) == "don’t 🙂 "  # "é" is cut: dropped
