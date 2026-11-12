"""Automatic labels for SAE concepts: prompts, scoring, batch clients and the site export.

``autointerp.py`` drives this. Every concept goes through two passes of the Message Batches API:

    label   the model sees the strongest examples and a few random ones, with the tokens the concept
            fires on marked, and returns a short label plus a one-sentence description
    score   the model sees only that label and description and a shuffled list of unmarked snippets,
            half of them held-out examples where the concept fires and half where it is silent, and
            says where it expects the concept to fire. The score is the balanced accuracy:
            0.5 is chance, 1.0 is perfect.

The examples a label is scored on are never shown to the labeling pass (``split_examples``).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from prizma_research.examples import Example, Examples

DEFAULT_MODEL = "claude-haiku-5-5"
# USD per million tokens through the Batch API (half the standard price), for prompts of up to 100K
# tokens. Checked on 2026-11-12: https://platform.claude.com/docs/en/about-claude/pricing
BATCH_PRICE_IN, BATCH_PRICE_OUT = 0.05, 0.25

N_TOP, N_RANDOM, N_HOLDOUT = 15, 5, 5  # shown strongest, shown random, held out for scoring
MIN_SHOWN, MIN_SCORED = 3, 3  # fewer examples than this: no label / no score
MARK_FRACTION = 0.3  # tokens at or above this share of the snippet's peak are marked
MAX_TOKENS = 1024  # thinking counts toward this; the answers themselves are well under 100 tokens
UNCLEAR = "unclear"

# For the cost estimate only. The current Claude tokenizer is not available offline; English text
# is roughly 3 characters per token with it, and snippets carry markers, so this errs on the high
# side. Output includes the (short, low effort) thinking the model does before it answers.
CHARS_PER_TOKEN = 2.8
EST_OUTPUT_TOKENS = {"label": 400, "score": 300}

LABEL_SYSTEM = """\
You are helping build Prizma, a website that lets curious people look inside a small language model (GPT-2 small). A sparse autoencoder has split the model's internal state into thousands of "concepts". Each concept is active ("fires") on a small share of the tokens the model reads and is silent on the rest. Your job is to work out what one concept responds to and to name it so that a visitor with no machine-learning background understands it at a glance.

You will get text snippets in which the concept fires. Tokens on which it fires are wrapped in double angle brackets, <<like this>>. The model reads left to right, so the concept can only depend on the marked token and the text before it. The text after a marked token is shown anyway, because some concepts are about what the model expects to come next. Each snippet starts with how strongly the concept fires in it, as a percentage of the strongest example. Snippets are cut out of web pages, so they begin and end mid-sentence. ⏎ stands for a line break and <|endoftext|> for the boundary between two unrelated pages.

How to read the evidence:
- Look first at what the marked tokens have in common: the same word or part of a word, a kind of word (months, surnames, units), a position (the start of a quote, after a comma in a list), a topic, or a tone.
- Then look at the text before them. If the marked tokens vary but the surrounding topic stays the same, the concept is about the topic.
- The strongest examples show the concept most clearly, and the random examples show how far it reaches. A good name covers both. If the random examples do not fit what the strongest ones share, name what the strongest ones share and mention the rest in the description.
- Choose the most specific description that still fits the examples. "the word 'bank' in a financial sense" is better than "finance", and "closing quotation marks" is better than "punctuation".

Your answer has two parts:
- label: at most 6 words, lower case except for proper names, no period at the end, and not starting with "concept" or "feature". It names the pattern itself, for example: sports scores, "un-" at the start of a word, legal language in contracts, years in the 1800s.
- description: one plain sentence of at most 30 words that says when the concept fires.

If you cannot find a pattern that most snippets share, use the label "unclear" and say in the description what you see instead."""

SCORE_SYSTEM = """\
You are helping check the names given to "concepts" inside a small language model (GPT-2 small). Each concept fires on some tokens of a text and is silent on most. Someone has named one concept and described when it fires. To test whether that name is accurate, you get the name, the description and a numbered list of short text snippets. In some snippets the concept fires on at least one token; in the others it does not fire at all. Going only by the name and the description, decide for each snippet whether the concept fires somewhere in it.

Snippets are cut out of web pages, so they begin and end mid-sentence. ⏎ stands for a line break and <|endoftext|> for the boundary between two unrelated pages. Judge every snippet on its own: the number of snippets in which the concept fires can be anything from none to all of them.

Answer with the numbers of the snippets in which you expect the concept to fire."""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
    "required": ["label", "description"],
    "additionalProperties": False,
}
SCORE_SCHEMA = {
    "type": "object",
    "properties": {"fires": {"type": "array", "items": {"type": "integer"}}},
    "required": ["fires"],
    "additionalProperties": False,
}


# ---- examples -> prompts ----------------------------------------------------------------------


def render(example: Example, *, mark: bool) -> str:
    """One snippet on one line; with ``mark`` the firing tokens are wrapped as ``<<token>>``."""
    threshold = MARK_FRACTION * example.peak
    parts = []
    for token, act in zip(example.tokens, example.acts):
        token = token.replace("\n", "⏎").replace("\r", "")
        if mark and act > 0 and act >= threshold:
            core = token.lstrip(" ")
            token = f"{token[:len(token) - len(core)]}<<{core}>>" if core else f"<<{token}>>"
        parts.append(token)
    return "".join(parts).strip()


@dataclass
class Split:
    top: list[Example]  # shown to the labeling pass, strongest first
    random: list[Example]  # shown to the labeling pass: random contexts where the concept fires
    holdout: list[Example]  # only ever seen by the scoring pass


def split_examples(examples: Examples, feature: int) -> Split:
    top = examples.top(feature)
    in_top = {e.context for e in top}
    extra = [e for e in examples.sample(feature) if e.context not in in_top]  # already in random order
    holdout, shown_random, shown_top = extra[:N_HOLDOUT], extra[N_HOLDOUT:N_HOLDOUT + N_RANDOM], top[:N_TOP]
    if len(holdout) < N_HOLDOUT:
        # A rare concept: (nearly) every context it fires on is in the top list. Hold out the ones the
        # labeling pass would not see anyway, then every third of the rest, strongest two kept.
        candidates = top[N_TOP:] + shown_top[2::3]
        taken = candidates[:N_HOLDOUT - len(holdout)]
        remaining = [e for e in shown_top if all(e is not t for t in taken)]
        if len(remaining) >= MIN_SHOWN and len(holdout) + len(taken) >= MIN_SCORED:
            holdout, shown_top = holdout + taken, remaining
    return Split(shown_top, shown_random, holdout)


def describe_density(density: float) -> str:
    if density <= 0:
        return "It never fired on the text that was scanned."
    return f"It fires on about 1 in {max(round(1 / density), 1):,} tokens."


def _request(custom_id: str, model: str, effort: str, system: str, user: str, schema: dict) -> dict:
    """A Message Batches request. No sampling parameters and no prefill: current models reject them."""
    return {"custom_id": custom_id, "params": {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    }}


def label_request(examples: Examples, feature: int, *, model: str = DEFAULT_MODEL, effort: str = "low") -> dict | None:
    """The labeling request for one concept, or None when it has too few examples to label."""
    split = split_examples(examples, feature)
    shown = split.top + split.random
    if len(shown) < MIN_SHOWN:
        return None
    strongest = max(e.peak for e in shown)

    def line(i: int, e: Example) -> str:
        return f"{i}. [{max(round(100 * e.peak / strongest), 1)}%] {render(e, mark=True)}"

    lines = [describe_density(examples.stats(feature)["density"]), "", "Strongest examples:"]
    lines += [line(i + 1, e) for i, e in enumerate(split.top)]
    if split.random:
        lines += ["", "Random examples where it fires:"]
        lines += [line(len(split.top) + i + 1, e) for i, e in enumerate(split.random)]
    return _request(f"label-{feature}", model, effort, LABEL_SYSTEM, "\n".join(lines), LABEL_SCHEMA)


def score_items(examples: Examples, feature: int, *, seed: int = 0) -> list[tuple[Example, bool]]:
    """Held-out firing examples and as many silent ones, shuffled. Empty when there are too few."""
    positives = split_examples(examples, feature).holdout
    negatives = examples.negatives(feature, len(positives), seed=seed)
    n = min(len(positives), len(negatives))
    if n < MIN_SCORED:
        return []
    items = [(e, True) for e in positives[:n]] + [(e, False) for e in negatives[:n]]
    random.Random(seed * 1_000_003 + feature).shuffle(items)
    return items


def score_request(items: list[tuple[Example, bool]], feature: int, label: str, description: str, *,
                  model: str = DEFAULT_MODEL, effort: str = "low") -> dict:
    lines = [f"Name: {label}", f"Description: {description}", "", "Snippets:"]
    lines += [f"{i + 1}. {render(e, mark=False)}" for i, (e, _) in enumerate(items)]
    return _request(f"score-{feature}", model, effort, SCORE_SYSTEM, "\n".join(lines), SCORE_SCHEMA)


# ---- answers -> rows --------------------------------------------------------------------------


def _answer(result: dict) -> tuple[dict | None, str]:
    """(parsed JSON, status). Status is ``ok`` or why there is no usable answer."""
    if result["type"] != "succeeded":
        return None, result["type"]  # errored, canceled or expired
    if result.get("stop_reason") == "refusal":
        return None, "refused"
    if result.get("stop_reason") == "max_tokens":
        return None, "truncated"
    try:
        data = json.loads(result.get("text") or "")
    except json.JSONDecodeError:
        return None, "invalid"
    return (data, "ok") if isinstance(data, dict) else (None, "invalid")


def parse_label(result: dict) -> dict:
    data, status = _answer(result)
    if status != "ok":
        return {"status": status, **({"error": result["error"]} if result.get("error") else {})}
    label = " ".join(str(data.get("label", "")).split()).rstrip(".")
    description = " ".join(str(data.get("description", "")).split())
    if not label:
        return {"status": "invalid"}
    return {"status": "ok", "label": label, "description": description}


def parse_score(result: dict, truth: list[bool]) -> dict:
    data, status = _answer(result)
    if status != "ok":
        return {"status": status}
    if not isinstance(data.get("fires"), list):
        return {"status": "invalid"}
    said = {int(i) for i in data["fires"] if isinstance(i, int)}
    tp = sum(1 for i, t in enumerate(truth, 1) if t and i in said)
    tn = sum(1 for i, t in enumerate(truth, 1) if not t and i not in said)
    n_pos = sum(truth)
    n_neg = len(truth) - n_pos
    score = (tp / n_pos + tn / n_neg) / 2
    return {"status": "ok", "score": round(score, 3), "tp": tp, "fn": n_pos - tp, "tn": tn, "fp": n_neg - tn}


def is_clear(row: dict | None) -> bool:
    """A label worth showing: the model answered and did not say ``unclear``."""
    return bool(row) and row.get("status") == "ok" and row["label"].lower() != UNCLEAR


def estimate_cost(requests: Iterable[dict], kind: str, *, price_in: float = BATCH_PRICE_IN,
                  price_out: float = BATCH_PRICE_OUT) -> dict:
    """Approximate cost of a list of requests (see ``CHARS_PER_TOKEN``)."""
    n = chars = 0
    for request in requests:
        params = request["params"]
        chars += len(params["system"]) + sum(len(m["content"]) for m in params["messages"])
        chars += len(json.dumps(params["output_config"]["format"]["schema"]))
        n += 1
    tokens_in, tokens_out = chars / CHARS_PER_TOKEN, n * EST_OUTPUT_TOKENS[kind]
    return {"requests": n, "input_tokens": round(tokens_in), "output_tokens": round(tokens_out),
            "usd": round((tokens_in * price_in + tokens_out * price_out) / 1e6, 4)}


# ---- batch clients ----------------------------------------------------------------------------


class AnthropicBatches:
    """The Message Batches API. Needs ``ANTHROPIC_API_KEY``; every submitted request is billed."""

    def __init__(self, client=None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def submit(self, requests: list[dict]) -> str:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        batch = self.client.messages.batches.create(requests=[
            Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"]))
            for r in requests])
        return batch.id

    def ended(self, batch_id: str) -> tuple[bool, str]:
        batch = self.client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        return batch.processing_status == "ended", (
            f"{c.processing} processing, {c.succeeded} succeeded, {c.errored} errored, {c.expired} expired")

    def results(self, batch_id: str) -> Iterator[dict]:
        for entry in self.client.messages.batches.results(batch_id):
            out = {"custom_id": entry.custom_id, "type": entry.result.type}
            if entry.result.type == "succeeded":
                message = entry.result.message
                out["text"] = next((b.text for b in message.content if b.type == "text"), None)  # thinking comes first
                out["stop_reason"] = message.stop_reason
                out["input_tokens"] = message.usage.input_tokens
                out["output_tokens"] = message.usage.output_tokens
            elif entry.result.type == "errored":
                out["error"] = f"{entry.result.error.error.type}: {entry.result.error.error.message}"
            yield out

    def count_tokens(self, request: dict) -> int:
        """Exact input tokens of one request. Free of charge, but needs the API key."""
        params = request["params"]
        return self.client.messages.count_tokens(
            model=params["model"], system=params["system"], messages=params["messages"]).input_tokens


class FakeBatches:
    """Offline stand-in for tests and ``--fake``: answers at once, from the prompt alone, for free.

    It labels a concept by the token that is marked most often and expects the concept to fire on
    every snippet that contains that token. Batches are kept on disk so that a second process can
    collect them, like real ones.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def submit(self, requests: list[dict]) -> str:
        self.folder.mkdir(parents=True, exist_ok=True)
        batch_id = f"fakebatch_{len(list(self.folder.glob('fakebatch_*.jsonl'))):04d}"
        with (self.folder / f"{batch_id}.jsonl").open("w") as f:
            for request in requests:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
        return batch_id

    def ended(self, batch_id: str) -> tuple[bool, str]:
        return True, "fake batch"

    def results(self, batch_id: str) -> Iterator[dict]:
        with (self.folder / f"{batch_id}.jsonl").open() as f:
            for line in f:
                request = json.loads(line)
                user = request["params"]["messages"][0]["content"]
                answer = self._label(user) if request["custom_id"].startswith("label-") else self._score(user)
                text = json.dumps(answer, ensure_ascii=False)
                yield {"custom_id": request["custom_id"], "type": "succeeded", "text": text, "stop_reason": "end_turn",
                       "input_tokens": round(len(user) / CHARS_PER_TOKEN), "output_tokens": round(len(text) / CHARS_PER_TOKEN)}

    @staticmethod
    def _label(user: str) -> dict:
        marked = Counter(re.findall(r"<<(.+?)>>", user))
        if not marked:
            return {"label": UNCLEAR, "description": "No token is marked."}
        token = marked.most_common(1)[0][0]
        return {"label": f'the token "{token}"', "description": f'Fires on the token "{token}".'}

    @staticmethod
    def _score(user: str) -> dict:
        match = re.search(r'^Name: the token "(.+)"$', user, flags=re.M)
        snippets = re.findall(r"^(\d+)\. (.*)$", user, flags=re.M)
        return {"fires": [int(i) for i, text in snippets if match and match.group(1) in text]}


# ---- site export ------------------------------------------------------------------------------


def export_site(examples: Examples, labels: dict[int, dict], scores: dict[int, dict], out: Path, *, source: str,
                shard_size: int = 32, n_examples: int = 8) -> dict:
    """Write ``labels.json`` and ``concepts/NNNN.json`` in the format the site reads.

    ``labels.json`` holds one label (or null) per concept id, plus the scores so the site can tell
    solid labels from weak ones without fetching the shards. Shard ``NNNN`` holds the concepts with
    ids ``[NNNN * shard_size, (NNNN + 1) * shard_size)``: label, description, score, statistics and
    the strongest examples. A concept without a usable label still gets its examples.
    """
    d_sae = examples.d_sae
    concepts = []
    for feature in range(d_sae):
        row, stats = labels.get(feature), examples.stats(feature)
        clear = is_clear(row)
        score = scores.get(feature, {}).get("score") if clear else None
        concepts.append({
            "id": feature,
            "label": row["label"] if clear else None,
            "description": row["description"] if row and row.get("status") == "ok" else None,
            "score": score,
            "max_act": round(stats["max_act"], 3),
            "density": stats["density"],
            "examples": [{"tokens": e.tokens, "acts": [round(a, 2) for a in e.acts]}
                         for e in examples.top(feature)[:n_examples]],
        })

    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps({
        "source": source,
        "labels": [c["label"] for c in concepts],
        "scores": [c["score"] for c in concepts],
    }, ensure_ascii=False) + "\n")
    shard_dir = out / "concepts"
    shard_dir.mkdir(exist_ok=True)
    for stale in shard_dir.glob("*.json"):
        stale.unlink()
    n_shards = -(-d_sae // shard_size)
    for shard in range(n_shards):
        chunk = concepts[shard * shard_size:(shard + 1) * shard_size]
        (shard_dir / f"{shard:04d}.json").write_text(
            json.dumps({"concepts": chunk}, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"concepts": d_sae, "labeled": sum(c["label"] is not None for c in concepts),
            "scored": sum(c["score"] is not None for c in concepts), "shards": n_shards, "shard_size": shard_size}


def write_review(examples: Examples, labels: dict[int, dict], scores: dict[int, dict], path: Path, *, n: int,
                 seed: int = 0) -> int:
    """A page of ``n`` random labeled concepts with their examples, for the human check of the
    Phase 1 gate ("most of 50 random concepts make sense")."""
    labeled = sorted(f for f, row in labels.items() if row.get("status") == "ok")
    picks = sorted(random.Random(seed).sample(labeled, min(n, len(labeled))))
    lines = [f"# Concept review: {len(picks)} random labeled concepts", "",
             "Read each label, then its examples (`<<token>>` = the concept fires). Does the label fit?", ""]
    for feature in picks:
        row, score = labels[feature], scores.get(feature, {}).get("score")
        lines += [f"## #{feature}: {row['label']}" + (f" (score {score:.2f})" if score is not None else ""), "",
                  f"{row['description']} {describe_density(examples.stats(feature)['density'])}", ""]
        lines += [f"- {render(e, mark=True)}" for e in examples.top(feature)[:6]]
        lines.append("")
    path.write_text("\n".join(lines))
    return len(picks)
