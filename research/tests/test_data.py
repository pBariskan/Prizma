"""The Hugging Face dataset path of the token stream, exercised on local parquet files (no network)."""

import itertools

import pytest

from prizma_research.data import iter_contexts
from prizma_research.lm import EOT_ID, LM, load_lm
from prizma_research.sae import SAE


@pytest.fixture(scope="module")
def real_mode_lm() -> LM:
    """The tiny model with ``random=False``: token streams then come from a dataset, as in the real run."""
    lm = load_lm(random=True, tiny=True)
    return LM(gpt=lm.gpt, tokenizer=lm.tokenizer, name="tiny", layer=lm.layer, random=False)


def write_parquet(path, column: str, rows: list) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({column: rows}), path)


def test_pretokenized_dataset_is_cut_into_bos_prefixed_contexts(real_mode_lm, tmp_path):
    """Same layout as the OpenWebText dataset: parquet shards with an ``input_ids`` column."""
    first = [[1000 * r + i for i in range(100)] for r in range(3)]
    second = [[7000 + 1000 * r + i for i in range(100)] for r in range(2)]
    write_parquet(tmp_path / "data" / "train-00000-of-00002.parquet", "input_ids", first)
    write_parquet(tmp_path / "data" / "train-00001-of-00002.parquet", "input_ids", second)

    contexts = list(iter_contexts(real_mode_lm, context_size=32, dataset=str(tmp_path)))
    assert len(contexts) == 5 * 3  # 100 tokens per row -> 3 pieces of 31, the last 7 tokens are dropped
    assert all(len(c) == 32 and c[0] == EOT_ID for c in contexts)
    assert contexts[0][1:] == list(range(31)) and contexts[1][1:] == list(range(31, 62))
    assert contexts[3][1:4] == [1000, 1001, 1002]  # the next row starts a new context

    held_out = list(iter_contexts(real_mode_lm, context_size=32, dataset=str(tmp_path),
                                  data_files="data/train-00001-of-00002.parquet"))
    assert len(held_out) == 2 * 3 and held_out[0][1] == 7000  # --data-files picks one shard


def test_text_dataset_is_tokenized_and_joined_with_eot(real_mode_lm, tmp_path):
    write_parquet(tmp_path / "data" / "train-00000-of-00001.parquet", "text", ["abc", "defgh", "ij" * 20])
    contexts = list(itertools.islice(iter_contexts(real_mode_lm, context_size=8, dataset=str(tmp_path)), 3))
    stream = [t for c in contexts for t in c[1:]]
    assert all(c[0] == EOT_ID and len(c) == 8 for c in contexts)
    assert stream[:10] == [*b"abc", EOT_ID, *b"defgh", EOT_ID]


def test_dataset_without_a_known_column_is_rejected(real_mode_lm, tmp_path):
    write_parquet(tmp_path / "data" / "train-00000-of-00001.parquet", "something", [1, 2])
    with pytest.raises(ValueError, match="input_ids"):
        next(iter_contexts(real_mode_lm, context_size=8, dataset=str(tmp_path)))


def test_open_sae_config_from_before_saelens_6():
    """The open res-jb SAEs ship an old ``cfg.json`` without architecture or input-space fields."""
    import torch

    d, n = 16, 40
    weights = {"W_enc": torch.randn(d, n), "b_enc": torch.zeros(n), "W_dec": torch.randn(n, d), "b_dec": torch.zeros(d)}
    old_cfg = {"model_name": "gpt2-small", "hook_point": "blocks.6.hook_resid_pre", "hook_point_layer": 6,
               "d_in": d, "d_sae": n, "context_size": 128, "dtype": "torch.float32", "l1_coefficient": 8e-05}
    plain = SAE.from_weights(weights, old_cfg, name="old")
    assert plain.architecture == "standard" and plain.k is None and plain.apply_b_dec_to_input and plain.prepend_bos
    assert plain.center_input is False
    # SAE.jbloom passes center_input=True: SAELens' registry says these SAEs read the centered residual.
    centered = SAE.from_weights(weights, old_cfg, name="old", center_input=True)
    x = torch.randn(5, d)
    assert torch.allclose(centered.encode(x), centered.encode(x + 3.0), atol=1e-5)  # a constant shift is removed
    assert not torch.allclose(plain.encode(x), plain.encode(x + 3.0), atol=1e-5)
    with pytest.raises(NotImplementedError):
        SAE.from_weights(weights, {**old_cfg, "normalize_activations": "expected_average_only_in"}, name="old")
