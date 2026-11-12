"""Label every SAE concept with Claude Haiku through the Batch API, score the labels, export for the site.

    uv run python autointerp.py --run runs/gpt2-l6-topk16384-k32                  # dry run: prints the cost, sends nothing
    uv run python autointerp.py --run runs/gpt2-l6-topk16384-k32 --sample 200 --submit   # pilot on 200 random concepts
    uv run python autointerp.py --run runs/gpt2-l6-topk16384-k32 --submit         # everything; costs money
    uv run python autointerp.py --run runs/<name> --fake                          # offline test, no API

Needs ``<run>/examples`` from ``collect_top.py``. Nothing is sent to the API without ``--submit``,
which needs ``ANTHROPIC_API_KEY`` and is billed. Three stages, each of which picks up where an
earlier call stopped (re-running never sends a concept twice):

    label    one request per concept: examples with the firing tokens marked -> label + description
    score    one request per labeled concept: label + unmarked held-out snippets -> where does it
             fire? Score = balanced accuracy (0.5 chance, 1.0 perfect)
    export   <run>/web/sae/labels.json and concepts/NNNN.json for the site, plus
             <run>/autointerp/review.md: 50 random concepts to read for the Phase 1 gate

State lives in ``<run>/autointerp/``: ``state.json`` (batch ids, token usage), ``labels.jsonl``,
``scores.jsonl``. Prompts and scoring: ``prizma_research/interp.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from prizma_research.common import write_json
from prizma_research.examples import Examples
from prizma_research.interp import (
    BATCH_PRICE_IN,
    BATCH_PRICE_OUT,
    DEFAULT_MODEL,
    AnthropicBatches,
    FakeBatches,
    estimate_cost,
    export_site,
    is_clear,
    label_request,
    parse_label,
    parse_score,
    score_items,
    score_request,
    write_review,
)

FINAL = ("ok", "refused")  # anything else (errored, expired, truncated, ...) is sent again by --retry-failed
PLACEHOLDER = ("a placeholder label of usual length", "A placeholder sentence of about the length that a real "
               "description has, used only to estimate the cost of scoring before the labels exist.")


class Store:
    """``<run>/autointerp``: batch ids and usage in ``state.json``, results in ``<kind>s.jsonl``."""

    def __init__(self, folder: Path, *, model: str, seed: int) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.state_path = folder / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
        else:
            self.state = {"model": model, "seed": seed, "batches": [], "usage": {}}

    def save(self) -> None:
        write_json(self.state_path, self.state)

    def rows(self, kind: str) -> dict[int, dict]:
        """Latest row per concept id."""
        path = self.folder / f"{kind}s.jsonl"
        rows = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                rows[row["id"]] = row
        return rows

    def append(self, kind: str, rows: list[dict]) -> None:
        with (self.folder / f"{kind}s.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def pending(self, kind: str) -> list[dict]:
        return [b for b in self.state["batches"] if b["kind"] == kind and not b["collected"]]

    def add_usage(self, kind: str, requests: int, tokens_in: int, tokens_out: int) -> None:
        usage = self.state["usage"].setdefault(kind, {"requests": 0, "input_tokens": 0, "output_tokens": 0})
        usage["requests"] += requests
        usage["input_tokens"] += tokens_in
        usage["output_tokens"] += tokens_out


def select_features(spec: str | None, sample: int | None, d_sae: int, seed: int) -> list[int]:
    features = list(range(d_sae))
    if spec:
        features = []
        for part in spec.split(","):
            lo, _, hi = part.partition("-")
            features += range(int(lo), int(hi or lo) + 1)
        features = sorted({f for f in features if 0 <= f < d_sae})
    if sample is not None and sample < len(features):
        features = sorted(random.Random(seed).sample(features, sample))
    return features


def build_requests(kind: str, todo: list[int], examples: Examples, store: Store, *, effort: str,
                   placeholder: bool) -> list[dict]:
    """Requests for the concepts in ``todo`` that can be labeled / scored (the rest are left out)."""
    model, seed = store.state["model"], store.state["seed"]
    if kind == "label":
        requests = (label_request(examples, f, model=model, effort=effort) for f in todo)
        return [r for r in requests if r is not None]
    labels = store.rows("label")
    requests = []
    for f in todo:
        row = labels.get(f)
        if is_clear(row):
            label, description = row["label"], row["description"]
        elif row is None and placeholder and label_request(examples, f) is not None:
            label, description = PLACEHOLDER  # dry run before labeling: estimate with a stand-in
        else:
            continue
        items = score_items(examples, f, seed=seed)
        if items:
            requests.append(score_request(items, f, label, description, model=model, effort=effort))
    return requests


def collect(kind: str, examples: Examples, store: Store, client, *, wait: bool, poll_seconds: float) -> bool:
    """Fetch the results of every submitted batch of this pass. False if one is still running."""
    for batch in store.pending(kind):
        while True:
            ended, counts = client.ended(batch["id"])
            if ended:
                break
            if not wait:
                print(f"[{kind}] batch {batch['id']} is still running ({counts}). Run the same command again later.")
                return False
            print(f"[{kind}] batch {batch['id']}: {counts}. Checking again in {poll_seconds:.0f} s.", flush=True)
            time.sleep(poll_seconds)

        rows, tokens_in, tokens_out = {}, 0, 0
        for result in client.results(batch["id"]):  # arrive in any order; the custom id says which concept
            feature = int(result["custom_id"].rsplit("-", 1)[1])
            if kind == "label":
                row = parse_label(result)
            else:
                row = parse_score(result, [fires for _, fires in score_items(examples, feature, seed=store.state["seed"])])
            rows[feature] = {"id": feature, **row}
            tokens_in += result.get("input_tokens", 0)
            tokens_out += result.get("output_tokens", 0)
        for feature in batch["features"]:
            rows.setdefault(feature, {"id": feature, "status": "missing"})
        store.append(kind, list(rows.values()))
        store.add_usage(kind, len(rows), tokens_in, tokens_out)
        batch["collected"] = True
        store.save()
        statuses = Counter(row["status"] for row in rows.values())
        print(f"[{kind}] batch {batch['id']} collected: " + ", ".join(f"{n} {s}" for s, n in statuses.most_common()))
    return True


def run_pass(kind: str, features: list[int], examples: Examples, store: Store, client, args, counter=None) -> bool:
    """One pass over the selected concepts. True when nothing is left to send or to wait for.

    ``client`` is None in a dry run; ``counter`` is an API client used only to count tokens.
    """
    if client is not None and not collect(kind, examples, store, client, wait=not args.no_wait,
                                          poll_seconds=args.poll_seconds):
        return False
    rows = store.rows(kind)
    in_flight = {f for batch in store.pending(kind) for f in batch["features"]}
    todo = [f for f in features if f not in in_flight
            and (f not in rows or (args.retry_failed and rows[f]["status"] not in FINAL))]
    requests = build_requests(kind, todo, examples, store, effort=args.effort, placeholder=client is None)
    done = sum(1 for f in features if f in rows)
    if not requests:
        waiting = f"{len(in_flight)} are in batches that are still running, " if in_flight else ""
        print(f"[{kind}] nothing to send: {done} of {len(features)} selected concepts have a result, {waiting}"
              f"the others have too few examples{'' if kind == 'label' else ' or no clear label'}.")
        return not in_flight

    estimate = estimate_cost(requests, kind, price_in=args.price_in, price_out=args.price_out)
    exact = ""
    if counter is not None:
        picks = random.Random(0).sample(requests, min(20, len(requests)))
        tokens_in = round(sum(counter.count_tokens(r) for r in picks) / len(picks) * len(requests))
        estimate["usd"] = round((tokens_in * args.price_in + estimate["output_tokens"] * args.price_out) / 1e6, 4)
        estimate["input_tokens"], exact = tokens_in, " (input counted by the API on a sample of 20)"
    print(f"[{kind}] {estimate['requests']} requests to send, about {estimate['input_tokens'] / 1e6:.2f}M input and "
          f"{estimate['output_tokens'] / 1e6:.2f}M output tokens: ${estimate['usd']:.2f}{exact}")

    if client is None:
        sample_path = store.folder / f"{kind}_requests.sample.jsonl"
        sample_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests[:5]))
        print(f"[{kind}] dry run: nothing sent. The first requests are in {sample_path}.")
        return False
    if estimate["usd"] > args.max_cost:
        raise SystemExit(f"[{kind}] estimated ${estimate['usd']:.2f} is above --max-cost ${args.max_cost:.2f}; "
                         "nothing sent. Raise --max-cost or narrow the selection with --sample / --features.")
    for start in range(0, len(requests), args.batch_size):
        chunk = requests[start:start + args.batch_size]
        batch_id = client.submit(chunk)
        store.state["batches"].append({"id": batch_id, "kind": kind, "collected": False,
                                       "features": [int(r["custom_id"].rsplit("-", 1)[1]) for r in chunk]})
        store.save()  # at once: a batch id that is lost is paid for twice
        print(f"[{kind}] submitted batch {batch_id} with {len(chunk)} requests", flush=True)
    return collect(kind, examples, store, client, wait=not args.no_wait, poll_seconds=args.poll_seconds)


def export(examples: Examples, store: Store, web_out: Path, args) -> dict:
    labels, scores = store.rows("label"), store.rows("score")
    ok_scores = {f: row for f, row in scores.items() if row["status"] == "ok"}
    model = store.state["model"]
    stats = export_site(examples, labels, ok_scores, web_out, shard_size=args.shard_size, n_examples=args.export_examples,
                        source=f"Prizma autointerp ({model}); score = balanced accuracy on held-out examples")
    n_review = write_review(examples, labels, ok_scores, store.folder / "review.md", n=args.review,
                            seed=store.state["seed"])

    values = [row["score"] for f, row in ok_scores.items() if is_clear(labels.get(f))]
    usage = store.state["usage"]
    tokens_in = sum(u["input_tokens"] for u in usage.values())
    tokens_out = sum(u["output_tokens"] for u in usage.values())
    summary = {
        **stats,
        "model": model,
        "label_status": dict(Counter(row["status"] for row in labels.values())),
        "unclear": sum(1 for row in labels.values() if row["status"] == "ok" and not is_clear(row)),
        "mean_score": round(sum(values) / len(values), 3) if values else None,
        "share_score_at_least_0.8": round(sum(v >= 0.8 for v in values) / len(values), 3) if values else None,
        "usage": usage,
        "cost_usd": round((tokens_in * args.price_in + tokens_out * args.price_out) / 1e6, 4),
    }
    write_json(store.folder / "summary.json", summary)
    print(f"[export] {stats['labeled']} of {stats['concepts']} concepts labeled, {stats['scored']} scored"
          + (f", mean score {summary['mean_score']:.2f}" if values else "")
          + f". Wrote {web_out} ({stats['shards']} shards) and {n_review} concepts to {store.folder / 'review.md'}.")
    if tokens_in:
        print(f"[export] API usage so far: {tokens_in / 1e6:.2f}M input, {tokens_out / 1e6:.2f}M output tokens, "
              f"${summary['cost_usd']:.2f} at batch prices")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="SAE run folder from train_sae.py")
    ap.add_argument("--examples", type=Path, default=None, help="examples folder (default: <run>/examples)")
    ap.add_argument("--out", type=Path, default=None, help="state and results (default: <run>/autointerp)")
    ap.add_argument("--web-out", type=Path, default=None, help="site files (default: <run>/web/sae)")
    ap.add_argument("--stage", choices=["all", "label", "score", "export"], default="all")
    ap.add_argument("--submit", action="store_true", help="send the batches: needs ANTHROPIC_API_KEY, costs money")
    ap.add_argument("--fake", action="store_true", help="offline stand-in instead of the API (tests)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="low", help="thinking effort: low, medium or high")
    ap.add_argument("--features", default=None, help="concept ids, e.g. '0-99,512' (default: all)")
    ap.add_argument("--sample", type=int, default=None, help="only this many random concepts of the selection")
    ap.add_argument("--batch-size", type=int, default=10_000, help="requests per batch (API limit: 100,000 or 256 MB)")
    ap.add_argument("--max-cost", type=float, default=15.0, help="refuse to send a pass estimated above this many USD")
    ap.add_argument("--price-in", type=float, default=BATCH_PRICE_IN, help="USD per million input tokens (batch)")
    ap.add_argument("--price-out", type=float, default=BATCH_PRICE_OUT, help="USD per million output tokens (batch)")
    ap.add_argument("--count-tokens", action="store_true",
                    help="estimate input tokens with the API's token counter (free, needs ANTHROPIC_API_KEY)")
    ap.add_argument("--no-wait", action="store_true", help="submit and exit; run again later to collect")
    ap.add_argument("--retry-failed", action="store_true", help="send concepts again whose request failed")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--export-examples", type=int, default=8, help="examples per concept in the site files")
    ap.add_argument("--shard-size", type=int, default=32, help="concepts per concepts/NNNN.json")
    ap.add_argument("--review", type=int, default=50, help="random labeled concepts in review.md")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.submit and args.fake:
        ap.error("--submit and --fake exclude each other")
    if args.submit and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--submit needs ANTHROPIC_API_KEY in the environment")

    examples = Examples.load(args.examples or args.run / "examples")
    store = Store(args.out or args.run / "autointerp", model="fake" if args.fake else args.model, seed=args.seed)
    features = select_features(args.features, args.sample, examples.d_sae, args.seed)
    client = FakeBatches(store.folder / "fake") if args.fake else AnthropicBatches() if args.submit else None
    counter = None
    if args.count_tokens and not args.fake:
        counter = client or AnthropicBatches()
    if store.state["model"] != ("fake" if args.fake else args.model):
        print(f"Note: {store.state_path} was started with model {store.state['model']}; keeping it.")
    print(f"{len(features)} of {examples.d_sae} concepts selected; model {store.state['model']}; "
          + ("offline fake" if args.fake else "sending to the API" if args.submit else "dry run, nothing is sent"))

    for stage in ["label", "score", "export"] if args.stage == "all" else [args.stage]:
        if stage == "export":
            if client is not None or args.stage == "export":
                export(examples, store, args.web_out or args.run / "web" / "sae", args)
        elif not run_pass(stage, features, examples, store, client, args, counter) and client is not None:
            print(f"[{stage}] not finished; later stages wait for it.")
            return 0
    if client is None and args.stage != "export":
        print("Dry run finished. Add --submit to send the requests (billed to your Anthropic account).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
