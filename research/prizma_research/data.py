"""Token streams for the Phase 1 pipeline.

Every script consumes the same thing: contexts of exactly ``context_size`` GPT-2 token ids that
start with ``<|endoftext|>``, because that is what the browser feeds the model (BOS + the prompt).

Sources:
    * a Hugging Face dataset (streamed), either pre-tokenized (``input_ids`` / ``tokens`` column)
      or raw text (``text`` column). The default is OpenWebText pre-tokenized for GPT-2.
    * ``--random``: endless synthetic English-looking text, tokenized one byte per token, so the
      whole pipeline runs offline.
"""

from __future__ import annotations

import argparse
import random as pyrandom
from collections.abc import Iterable, Iterator

import torch

from prizma_research.lm import EOT_ID, LM, Tokenizer

# 8.8M rows of 1,024 GPT-2 tokens (about 9B tokens) in 73 parquet shards of ~124M tokens each.
OPENWEBTEXT = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
# Training streams from the first shard (200M tokens end inside the second one), so the last
# shards are held out. Evaluation and example collection read different ones.
EVAL_DATA_FILES = "data/train-00072-of-00073.parquet"
EXAMPLES_DATA_FILES = "data/train-00071-of-00073.parquet"

_SUBJECTS = ["The cat", "My sister", "A young doctor", "The old captain", "Our teacher", "The president",
             "A tired farmer", "The little robot", "His neighbour", "The chef"]
_VERBS = ["bought", "painted", "found", "forgot", "repaired", "described", "cooked", "counted", "sold", "loved"]
_OBJECTS = ["a red bicycle", "three green apples", "the wooden boat", "an old piano", "two silver coins",
            "a map of Paris", "the broken clock", "some fresh bread", "a letter from London", "the last ticket"]
_TAILS = ["in the morning", "near the river", "before the wedding", "during the storm", "for 20 dollars",
          "in March 1969", "without a word", "at the market", "after midnight", "with great care"]


def synthetic_documents(seed: int = 0) -> Iterator[str]:
    """Endless stream of short template sentences (offline stand-in for OpenWebText)."""
    rng = pyrandom.Random(seed)
    while True:
        n = rng.randint(3, 8)
        yield " ".join(
            f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_TAILS)}."
            for _ in range(n)
        )


def _contexts_from_rows(rows: Iterable[list[int]], context_size: int) -> Iterator[list[int]]:
    """Pre-tokenized rows: cut each row into pieces and put BOS in front. The row's tail is dropped."""
    n = context_size - 1
    for row in rows:
        for start in range(0, len(row) - n + 1, n):
            yield [EOT_ID, *row[start:start + n]]


def _contexts_from_documents(docs: Iterable[list[int]], context_size: int) -> Iterator[list[int]]:
    """Tokenized documents of any length: join them with <|endoftext|> and cut the stream into pieces."""
    n = context_size - 1
    buffer: list[int] = []
    for doc in docs:
        buffer.extend(doc)
        buffer.append(EOT_ID)
        while len(buffer) >= n:
            yield [EOT_ID, *buffer[:n]]
            del buffer[:n]


def _hub_rows(dataset: str, data_files: str | None) -> Iterator[dict]:
    from datasets import load_dataset

    yield from load_dataset(dataset, split="train", streaming=True, data_files=data_files)


def _check_context_size(lm: LM, context_size: int) -> None:
    if context_size > lm.gpt.cfg.n_positions:
        raise ValueError(f"context_size {context_size} exceeds the model's {lm.gpt.cfg.n_positions} positions")


def iter_contexts(lm: LM, *, context_size: int, dataset: str = OPENWEBTEXT, data_files: str | None = None,
                  seed: int = 0) -> Iterator[list[int]]:
    """Yield contexts of exactly ``context_size`` token ids, each starting with <|endoftext|>."""
    _check_context_size(lm, context_size)
    yield from _iter_contexts(lm.tokenizer, lm.random, context_size, dataset, data_files, seed)


def _iter_contexts(tokenizer: Tokenizer, synthetic: bool, context_size: int, dataset: str, data_files: str | None,
                   seed: int) -> Iterator[list[int]]:
    if synthetic:
        docs = (tokenizer.encode(text) for text in synthetic_documents(seed))
        yield from _contexts_from_documents(docs, context_size)
        return

    rows = _hub_rows(dataset, data_files)
    first = next(rows, None)
    if first is None:
        raise ValueError(f"dataset {dataset!r} (data_files={data_files!r}) is empty")
    column = next((c for c in ("input_ids", "tokens", "text") if c in first), None)
    if column is None:
        raise ValueError(f"dataset {dataset!r} needs an 'input_ids', 'tokens' or 'text' column, has {list(first)}")

    def values() -> Iterator:
        yield first[column]
        for row in rows:
            yield row[column]

    if column == "text":
        yield from _contexts_from_documents((tokenizer.encode(text) for text in values()), context_size)
    else:
        yield from _contexts_from_rows((list(row) for row in values()), context_size)


def iter_token_batches(contexts: Iterable[list[int]], *, batch_size: int, n_tokens: int,
                       device: torch.device | str = "cpu") -> Iterator[torch.Tensor]:
    """Batch contexts into [B, T] tensors until about ``n_tokens`` tokens have been yielded."""
    batch: list[list[int]] = []
    seen = 0
    for ctx in contexts:
        batch.append(ctx)
        seen += len(ctx)
        if len(batch) == batch_size or seen >= n_tokens:
            yield torch.tensor(batch, dtype=torch.long, device=device)
            batch = []
        if seen >= n_tokens:
            return
    if batch:
        yield torch.tensor(batch, dtype=torch.long, device=device)


def as_hf_dataset(lm: LM, *, context_size: int, dataset: str, data_files: str | None, seed: int):
    """The same contexts as an ``IterableDataset`` with an ``input_ids`` column, the form SAELens reads."""
    from datasets import Features, IterableDataset, Sequence, Value

    _check_context_size(lm, context_size)
    tokenizer, synthetic = lm.tokenizer, lm.random  # the generator must not hold the model: `datasets` hashes it

    def generate() -> Iterator[dict]:
        for ctx in _iter_contexts(tokenizer, synthetic, context_size, dataset, data_files, seed):
            yield {"input_ids": ctx}

    return IterableDataset.from_generator(generate, features=Features({"input_ids": Sequence(Value("int64"))}))


def add_data_args(ap: argparse.ArgumentParser, *, default_files: str | None) -> None:
    ap.add_argument("--dataset", default=OPENWEBTEXT,
                    help="Hugging Face dataset id or local folder; pre-tokenized (input_ids) or raw text")
    ap.add_argument("--data-files", default=default_files,
                    help=f"restrict the dataset to these files (default: {default_files or 'all, from the start'})")
