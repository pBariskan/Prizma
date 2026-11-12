"""The evaluation metrics on SAEs whose correct numbers are known in advance."""

import itertools
import json

import pytest
import torch

import eval as eval_script
from prizma_research.data import iter_contexts
from prizma_research.lm import EOT_ID, load_lm, logits_from_resid, resid_at
from prizma_research.metrics import Evaluator, gate, loss_sum
from prizma_research.sae import SAE, center


@pytest.fixture(scope="module")
def lm():
    return load_lm(random=True, tiny=True)


@pytest.fixture(scope="module")
def tokens(lm):
    return torch.tensor(list(itertools.islice(iter_contexts(lm, context_size=24, seed=5), 12)))


def exact_sae(d: int, **kw) -> SAE:
    """x = relu(x) - relu(-x): reconstructs any input without error."""
    eye = torch.eye(d)
    return SAE(W_enc=torch.cat([eye, -eye], dim=1), b_enc=torch.zeros(2 * d), W_dec=torch.cat([eye, -eye]),
               b_dec=torch.zeros(d), name="exact", **kw)


def silent_sae(d: int) -> SAE:
    """Never fires and reconstructs everything as zero."""
    return SAE(W_enc=torch.zeros(d, 8), b_enc=-torch.ones(8), W_dec=torch.zeros(8, d), b_dec=torch.zeros(d), name="silent")


def evaluate(lm, tokens, **saes) -> dict:
    evaluator = Evaluator(lm, saes)
    for batch in tokens.split(5):
        evaluator.add(batch, valid=batch != EOT_ID)
    return evaluator.result()


def test_exact_and_silent_sae(lm, tokens):
    d = lm.d_model
    result = evaluate(lm, tokens, exact=exact_sae(d), exact_centered=exact_sae(d, center_input=True), silent=silent_sae(d))
    assert result["n_tokens"] == tokens.numel() and result["n_valid_tokens"] == int((tokens != EOT_ID).sum())
    assert result["zero_ablation_loss"] != result["clean_loss"]
    for name in ("exact", "exact_centered"):
        m = result["saes"][name]
        assert m["explained_variance"] == pytest.approx(1, abs=1e-6)
        assert m["explained_variance_centered"] == pytest.approx(1, abs=1e-6)
        assert m["loss_increase"] == pytest.approx(0, abs=1e-4) and m["loss_recovered"] == pytest.approx(1, abs=1e-2)
        assert m["l0"] == pytest.approx(d, abs=1) and m["dead_fraction"] == 0 and m["dense_fraction"] == 1
    silent = result["saes"]["silent"]
    assert silent["l0"] == 0 and silent["dead_fraction"] == 1 and silent["dense_fraction"] == 0
    assert silent["explained_variance"] < 0.5  # only as good as predicting zero
    assert silent["spliced_loss"] == pytest.approx(result["zero_ablation_loss"], abs=1e-5)
    assert silent["loss_recovered"] == pytest.approx(0, abs=1e-3)


def test_metrics_match_a_direct_computation(lm, tokens):
    sae = SAE.random_init(lm.d_model, 96, architecture="topk", k=6, seed=3)
    m = evaluate(lm, tokens, sae=sae)["saes"]["sae"]

    valid = tokens != EOT_ID
    resid = resid_at(lm.gpt, tokens, lm.layer)
    x, x_hat = resid[valid], sae.reconstruct(resid)[valid]
    assert m["explained_variance"] == pytest.approx(
        float(1 - (x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum()), abs=1e-5)
    xc, ec = center(x), center(x - x_hat)
    assert m["explained_variance_centered"] == pytest.approx(
        float(1 - ec.pow(2).sum() / (xc - xc.mean(0)).pow(2).sum()), abs=1e-5)
    assert m["l0"] == pytest.approx(float((sae.encode(resid)[valid] > 0).sum(-1).float().mean()), abs=1e-5)
    spliced = torch.where(valid[..., None], sae.reconstruct(resid), resid)
    positions = valid[:, :-1]
    want = loss_sum(logits_from_resid(lm.gpt, spliced, lm.layer), tokens, positions) / int(positions.sum())
    assert m["spliced_loss"] == pytest.approx(want, abs=1e-5)


def test_gate():
    base = {"explained_variance_centered": 0.90, "loss_recovered": 0.99, "dead_fraction": 0.0}
    close = {"explained_variance_centered": 0.86, "loss_recovered": 0.975, "dead_fraction": 0.05}
    assert all(gate(close, base).values())
    assert gate({**close, "explained_variance_centered": 0.84}, base)["explained_variance"] is False
    assert gate({**close, "loss_recovered": 0.96}, base)["loss_recovered"] is False
    assert gate({**close, "dead_fraction": 0.2}, base)["dead_concepts"] is False


def test_eval_script_on_the_tiny_run(tiny_run, tmp_path):
    assert eval_script.main(["--sae", str(tiny_run), "--random", "--tiny", "--device", "cpu", "--out", str(tmp_path),
                             "--n-tokens", "6000"]) == 0
    result = json.loads((tmp_path / "eval.json").read_text())
    assert result["sae"] == "tiny" and result["baseline"] == "untrained-sae" and set(result["gate"]["checks"]) == {
        "explained_variance", "loss_recovered", "dead_concepts"}
    ours, baseline = result["saes"]["tiny"], result["saes"]["untrained-sae"]
    assert ours["explained_variance"] > 0.8 > baseline["explained_variance"]
    assert 7 <= ours["l0"] <= 8 and ours["k"] == 8 and baseline["center_input"] is True
    report = (tmp_path / "eval.md").read_text()
    assert "| Metric | tiny | untrained-sae |" in report and "Phase 1 gate" in report
