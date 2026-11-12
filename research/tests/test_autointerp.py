"""Labeling and scoring without the API: prompts, the example split, parsing, resuming, the site files."""

import json

import pytest
import torch

import autointerp
from prizma_research import interp
from prizma_research.examples import ExampleCollector, Examples

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def letter_examples(n_ctx: int = 400, T: int = 16) -> Examples:
    """Concept f fires on the letter with index f and nowhere else; concept 25 ('z') fires in 4 contexts only."""
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 25, (n_ctx, T), generator=g)
    tokens[:4, 5] = 25
    tokens[:, 0] = 50256
    acts = torch.nn.functional.one_hot(tokens.clamp(max=25), 26).float() * (1 + torch.rand(n_ctx, T, 1, generator=g))
    collector = ExampleCollector(26, T, pool_size=256)
    for i in range(0, n_ctx, 50):
        collector.add(tokens[i:i + 50], acts[i:i + 50], valid=tokens[i:i + 50] != 50256)
    vocab = {i: c for i, c in enumerate(LETTERS)} | {50256: "<|endoftext|>"}
    return Examples(collector.result(), vocab, {"d_sae": 26})


def test_scoring_examples_are_held_out():
    ex = letter_examples()
    for feature in range(25):
        split = interp.split_examples(ex, feature)
        shown = {e.context for e in split.top + split.random}
        assert len(split.top) == interp.N_TOP and len(split.holdout) == interp.N_HOLDOUT
        assert not shown & {e.context for e in split.holdout}
        items = interp.score_items(ex, feature, seed=0)
        assert sum(fires for _, fires in items) == len(items) / 2 == interp.N_HOLDOUT
        assert items == interp.score_items(ex, feature, seed=0)  # the same again when results are collected
        assert all((e.peak > 0) == fires for e, fires in items)
    # 'z' fires in 4 contexts: enough to label, too few to hold any out for scoring.
    rare = interp.split_examples(ex, 25)
    assert len(rare.top) == 4 and rare.holdout == [] and interp.score_items(ex, 25) == []
    assert interp.label_request(ex, 25) is not None


def test_requests_follow_the_current_api():
    ex = letter_examples()
    request = interp.label_request(ex, 3)
    params = request["params"]
    assert request["custom_id"] == "label-3" and params["model"] == "claude-haiku-5-5"
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert not {"temperature", "top_p", "top_k", "thinking", "fallbacks"} & set(params)
    assert params["messages"][-1]["role"] == "user"  # no prefill
    user = params["messages"][0]["content"]
    assert "<<d>>" in user and "1. [100%]" in user and "Random examples where it fires:" in user

    items = interp.score_items(ex, 3, seed=0)
    user = interp.score_request(items, 3, "the letter d", "Fires on d.")["params"]["messages"][0]["content"]
    assert "<<" not in user and "Name: the letter d" in user and f"{len(items)}. " in user


def test_parse_answers():
    ok = {"type": "succeeded", "stop_reason": "end_turn"}
    assert interp.parse_label({**ok, "text": '{"label": "Sports  scores.", "description": "Fires on scores."}'}) == {
        "status": "ok", "label": "Sports scores", "description": "Fires on scores."}
    assert interp.parse_label({"type": "succeeded", "stop_reason": "refusal", "text": None})["status"] == "refused"
    assert interp.parse_label({"type": "succeeded", "stop_reason": "max_tokens", "text": None})["status"] == "truncated"
    assert interp.parse_label({**ok, "text": "not json"})["status"] == "invalid"
    assert interp.parse_label({"type": "expired"})["status"] == "expired"
    assert interp.parse_label({"type": "errored", "error": "overloaded_error: busy"}) == {
        "status": "errored", "error": "overloaded_error: busy"}

    truth = [True, False, True, False, True, False]
    perfect = interp.parse_score({**ok, "text": '{"fires": [1, 3, 5]}'}, truth)
    assert perfect["score"] == 1.0 and perfect["tp"] == 3 and perfect["tn"] == 3
    assert interp.parse_score({**ok, "text": '{"fires": []}'}, truth)["score"] == 0.5  # says "never": chance
    assert interp.parse_score({**ok, "text": '{"fires": [1, 2, 3, 4, 5, 6]}'}, truth)["score"] == 0.5
    assert interp.parse_score({**ok, "text": '{"fires": [2, 4, 6]}'}, truth)["score"] == 0.0
    assert interp.parse_score({**ok, "text": '{"fires": [1, 2]}'}, truth)["score"] == pytest.approx(0.5, abs=1e-3)
    assert interp.parse_score({"type": "errored"}, truth) == {"status": "errored"}


def test_fake_labels_score_perfectly_on_single_token_concepts(tmp_path):
    ex = letter_examples()
    ex.save(tmp_path / "examples")
    args = ["--run", str(tmp_path), "--fake", "--shard-size", "8", "--review", "5"]
    assert autointerp.main(args) == 0

    labels = json.loads((tmp_path / "web" / "sae" / "labels.json").read_text())
    assert labels["labels"] == [f'the token "{c}"' for c in LETTERS]
    assert labels["scores"] == [1.0] * 25 + [None]  # 'z' is too rare to score
    shards = sorted((tmp_path / "web" / "sae" / "concepts").glob("*.json"))
    assert [p.name for p in shards] == ["0000.json", "0001.json", "0002.json", "0003.json"]
    concepts = [c for p in shards for c in json.loads(p.read_text())["concepts"]]
    assert [c["id"] for c in concepts] == list(range(26))
    first = concepts[0]
    assert first["label"] == 'the token "a"' and first["score"] == 1.0 and 0 < first["density"] < 0.1
    assert len(first["examples"]) == 8 and len(first["examples"][0]["tokens"]) == len(first["examples"][0]["acts"])
    assert max(first["examples"][0]["acts"]) == pytest.approx(first["max_act"], abs=0.01)
    assert (tmp_path / "autointerp" / "review.md").read_text().count("\n## #") == 5

    # Running again sends nothing: every concept already has its result.
    state = json.loads((tmp_path / "autointerp" / "state.json").read_text())
    assert [b["kind"] for b in state["batches"]] == ["label", "score"] and state["usage"]["label"]["requests"] == 26
    assert autointerp.main(args) == 0
    assert json.loads((tmp_path / "autointerp" / "state.json").read_text()) == state


def test_dry_run_sends_and_stores_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    letter_examples().save(tmp_path / "examples")
    assert autointerp.main(["--run", str(tmp_path), "--features", "0-24", "--sample", "10"]) == 0
    out = capsys.readouterr().out
    assert "[label] 10 requests to send" in out and "[score] 10 requests to send" in out and "nothing sent" in out
    assert not (tmp_path / "autointerp" / "state.json").exists() and not (tmp_path / "web").exists()
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        autointerp.main(["--run", str(tmp_path), "--submit"])


def test_submitted_batches_are_collected_not_resent(tmp_path, monkeypatch):
    """A batch that is still running is left alone, and collected by the next call."""
    letter_examples().save(tmp_path / "examples")
    running = {"value": True}
    monkeypatch.setattr(interp.FakeBatches, "ended", lambda self, batch_id: (not running["value"], "1 processing"))
    args = ["--run", str(tmp_path), "--fake", "--no-wait", "--features", "0-9", "--batch-size", "4"]
    assert autointerp.main(args) == 0
    state = json.loads((tmp_path / "autointerp" / "state.json").read_text())
    assert [len(b["features"]) for b in state["batches"]] == [4, 4, 2] and not any(b["collected"] for b in state["batches"])
    assert autointerp.main(args) == 0  # still running: nothing new is submitted
    assert len(json.loads((tmp_path / "autointerp" / "state.json").read_text())["batches"]) == 3

    running["value"] = False
    assert autointerp.main(args) == 0
    state = json.loads((tmp_path / "autointerp" / "state.json").read_text())
    assert [b["kind"] for b in state["batches"]] == ["label"] * 3 + ["score"] * 3 and all(b["collected"] for b in state["batches"])
    labels = json.loads((tmp_path / "web" / "sae" / "labels.json").read_text())["labels"]
    assert labels[:10] == [f'the token "{c}"' for c in LETTERS[:10]] and labels[10:] == [None] * 16


def test_max_cost_guard(tmp_path):
    letter_examples().save(tmp_path / "examples")
    with pytest.raises(SystemExit, match="--max-cost"):
        autointerp.main(["--run", str(tmp_path), "--fake", "--max-cost", "0.0001"])
    assert not (tmp_path / "autointerp" / "state.json").exists()


def test_anthropic_client_wire_format():
    """Our requests and result parsing against the real SDK, with the HTTP layer replaced."""
    import anthropic
    import httpx2

    sent = {}
    batch = {"id": "msgbatch_1", "type": "message_batch", "processing_status": "ended",
             "created_at": "2026-11-12T00:00:00Z", "expires_at": "2026-11-13T00:00:00Z",
             "request_counts": {"processing": 0, "succeeded": 1, "errored": 1, "canceled": 0, "expired": 0},
             "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"}
    message = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
               "content": [{"type": "thinking", "thinking": "", "signature": "sig"},
                           {"type": "text", "text": '{"label": "the letter d", "description": "Fires on d."}'}],
               "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 900, "output_tokens": 120}}
    lines = [{"custom_id": "label-3", "result": {"type": "succeeded", "message": message}},
             {"custom_id": "label-4", "result": {"type": "errored", "error": {
                 "type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}}}]

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            sent.update(json.loads(request.content))
            return httpx2.Response(200, json=batch)
        if request.url.path.endswith("/results"):
            return httpx2.Response(200, content="\n".join(json.dumps(line) for line in lines))
        return httpx2.Response(200, json=batch)

    client = interp.AnthropicBatches(anthropic.Anthropic(
        api_key="test", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))))
    ex = letter_examples()
    assert client.submit([interp.label_request(ex, 3), interp.label_request(ex, 4)]) == "msgbatch_1"
    assert [r["custom_id"] for r in sent["requests"]] == ["label-3", "label-4"]
    params = sent["requests"][0]["params"]
    assert params["model"] == "claude-haiku-5-5" and params["max_tokens"] == 1024
    assert params["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": interp.LABEL_SCHEMA}}

    assert client.ended("msgbatch_1")[0] is True
    ok, failed = list(client.results("msgbatch_1"))
    assert interp.parse_label(ok) == {"status": "ok", "label": "the letter d", "description": "Fires on d."}
    assert (ok["input_tokens"], ok["output_tokens"]) == (900, 120)
    assert interp.parse_label(failed) == {"status": "errored", "error": "invalid_request_error: bad"}


def test_tiny_pipeline_labels(tiny_examples, tmp_path):
    """collect_top.py output of the tiny SAE goes through the whole labeling flow."""
    run = tiny_examples.parent
    assert autointerp.main(["--run", str(run), "--fake", "--out", str(tmp_path / "state"), "--web-out", str(tmp_path / "web")]) == 0
    summary = json.loads((tmp_path / "state" / "summary.json").read_text())
    d_sae = Examples.load(tiny_examples).d_sae
    assert summary["concepts"] == d_sae and 0 < summary["scored"] <= summary["labeled"] < d_sae
    assert 0.5 <= summary["mean_score"] <= 1.0
    labels = json.loads((tmp_path / "web" / "labels.json").read_text())
    assert len(labels["labels"]) == len(labels["scores"]) == d_sae
    assert len(list((tmp_path / "web" / "concepts").glob("*.json"))) == -(-d_sae // 32)
