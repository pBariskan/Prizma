"""The SAE web artifact must round-trip and encode the way the manifest says."""

import json
import sys

import numpy as np
import pytest
import torch

import export_sae
from prizma_research import sae_artifact as sae_format
from prizma_research.sae_artifact import SaeSettings

# cfg.json of jbloom/GPT2-Small-SAEs-Reformatted (the keys that matter; the rest is training metadata).
LEGACY_CFG = {"model_name": "gpt2-small", "hook_point": "blocks.6.hook_resid_pre", "d_in": 768, "d_sae": 24576}


def random_weights(d_model=16, d_sae=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "W_enc": torch.randn(d_model, d_sae, generator=g),
        "b_enc": torch.randn(d_sae, generator=g) * 0.1,
        "W_dec": torch.randn(d_sae, d_model, generator=g),
        "b_dec": torch.randn(d_model, generator=g) * 0.1,
    }


def test_legacy_cfg_gets_saelens_defaults():
    s = sae_format.settings_from_cfg(LEGACY_CFG)
    assert s == SaeSettings(activation="relu", k=None, center_resid=True, apply_b_dec_to_input=True, prepend_bos=True)


def test_saelens_v6_topk_cfg():
    cfg = {
        "architecture": "topk",
        "k": 32,
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "metadata": {"prepend_bos": True, "model_from_pretrained_kwargs": {"center_writing_weights": False}},
    }
    s = sae_format.settings_from_cfg(cfg)
    assert s == SaeSettings(activation="topk", k=32, center_resid=False, apply_b_dec_to_input=False, prepend_bos=True)
    # Older SAELens stored k under activation_fn_kwargs.
    old = {"architecture": "topk", "activation_fn_kwargs": {"k": 8}}
    assert sae_format.settings_from_cfg(old).k == 8


@pytest.mark.parametrize("cfg", [
    {"architecture": "jumprelu"},
    {"architecture": "topk"},
    {"normalize_activations": "expected_average_only_in"},
])
def test_unsupported_cfgs_are_rejected(cfg):
    with pytest.raises(ValueError):
        sae_format.settings_from_cfg(cfg)


def test_encode_relu_centering_and_b_dec():
    w = random_weights()
    x = torch.randn(5, 16)
    plain = SaeSettings("relu", None, center_resid=False, apply_b_dec_to_input=False, prepend_bos=True)
    assert torch.allclose(sae_format.encode(x, w, plain), (x @ w["W_enc"] + w["b_enc"]).relu())

    full = SaeSettings("relu", None, center_resid=True, apply_b_dec_to_input=True, prepend_bos=True)
    expected = ((x - x.mean(-1, keepdim=True) - w["b_dec"]) @ w["W_enc"] + w["b_enc"]).relu()
    assert torch.allclose(sae_format.encode(x, w, full), expected)
    # Centering makes the encoding blind to a constant offset, like the model's LayerNorms.
    assert torch.allclose(sae_format.encode(x + 3.0, w, full), expected, atol=1e-4)


def test_encode_topk_keeps_k_largest():
    w = random_weights()
    x = torch.randn(7, 16)
    s = SaeSettings("topk", 4, center_resid=False, apply_b_dec_to_input=True, prepend_bos=True)
    acts = sae_format.encode(x, w, s)
    pre = (x - w["b_dec"]) @ w["W_enc"] + w["b_enc"]
    assert ((acts > 0).sum(-1) <= 4).all()
    for row_acts, row_pre in zip(acts, pre):
        top = row_pre.topk(4)
        assert torch.allclose(row_acts[top.indices], top.values.relu())


def test_weights_round_trip_through_files(tmp_path):
    w = random_weights(d_model=16, d_sae=40)
    files = sae_format.write_weights(tmp_path, w, shard_size=16)
    assert sorted(p.name for p in (tmp_path / "decoder").iterdir()) == ["0000.f16.bin", "0001.f16.bin", "0002.f16.bin"]
    assert (tmp_path / "decoder" / "0002.f16.bin").stat().st_size == 8 * 16 * 2  # last shard is short
    assert files["bytes"]["encoder.f16.bin"] == 40 * 16 * 2

    stored = sae_format.read_weights(tmp_path, files)
    for key in w:
        assert stored[key].shape == w[key].shape
        assert torch.allclose(stored[key], w[key], atol=2e-3, rtol=1e-3), key
    # Row f of the encoder file is the encoder weight of feature f.
    enc = np.frombuffer((tmp_path / "encoder.f16.bin").read_bytes(), dtype="<f2").reshape(40, 16)
    assert np.allclose(enc[7], w["W_enc"][:, 7].numpy(), atol=2e-3)

    # A smaller SAE written to the same folder leaves no stale shards, and gets a new version.
    files_small = sae_format.write_weights(tmp_path, random_weights(d_model=16, d_sae=16, seed=1), shard_size=16)
    assert [p.name for p in (tmp_path / "decoder").iterdir()] == ["0000.f16.bin"]
    assert files_small["version"] != files["version"]


def test_fp16_overflow_is_an_error(tmp_path):
    w = random_weights()
    w["W_enc"][0, 0] = 1e6
    with pytest.raises(ValueError):
        sae_format.write_weights(tmp_path, w)


def test_selfcheck_matches_encode():
    w = random_weights()
    s = SaeSettings("relu", None, center_resid=True, apply_b_dec_to_input=True, prepend_bos=True)
    check = sae_format.selfcheck(torch.randn(16), w, s)
    acts = sae_format.encode(torch.tensor(check["resid"]), w, s)
    assert check["features"] == acts.topk(10).indices.tolist()
    assert np.allclose(check["acts"], acts.topk(10).values.numpy())


@pytest.mark.parametrize("architecture", ["standard", "topk"])
def test_export_sae_random_end_to_end(tmp_path, monkeypatch, architecture):
    """export_sae.py --random on a tiny random model folder (only the model manifest is needed)."""
    model_dir = tmp_path / "gpt2-random-tiny"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(json.dumps({
        "name": "gpt2-random-tiny", "source": "random", "split_layer": 2, "hook": "blocks.2.hook_resid_pre",
        "n_layer": 4, "n_head": 4, "d_model": 64, "vocab_size": 50257,
    }))
    argv = ["export_sae.py", "--random", "--model-dir", str(model_dir), "--d-sae", "100", "--shard-size", "32",
            "--architecture", architecture, "--k", "5"]
    monkeypatch.setattr(sys, "argv", argv)
    assert export_sae.main() == 0

    manifest, settings = sae_format.read_manifest(model_dir / "sae")
    assert manifest["d_sae"] == 100 and manifest["d_model"] == 64 and manifest["hook"] == "blocks.2.hook_resid_pre"
    assert manifest["decoder_shard_size"] == 32 and len(list((model_dir / "sae" / "decoder").iterdir())) == 4
    assert settings.activation == ("topk" if architecture == "topk" else "relu")
    assert manifest.get("k") == (5 if architecture == "topk" else None)
    assert not settings.center_resid and settings.prepend_bos and manifest["bos_token_id"] == 50256
    assert "labels" not in manifest["files"]

    # The self-check stored in the manifest is reproducible from the stored files.
    stored = sae_format.read_weights(model_dir / "sae", manifest)
    acts = sae_format.encode(torch.tensor(manifest["selfcheck"]["resid"]), stored, settings)
    assert np.allclose(acts[manifest["selfcheck"]["features"]].numpy(), manifest["selfcheck"]["acts"], rtol=1e-4, atol=1e-5)
    if architecture == "topk":
        assert manifest["verification"]["mean_l0"] <= 5


def test_export_fails_below_the_minimum_explained_variance(tmp_path, monkeypatch, capsys):
    """A random SAE explains nothing: with --min-ev the export fails and leaves no manifest behind."""
    model_dir = tmp_path / "gpt2-random-tiny"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(json.dumps({
        "name": "gpt2-random-tiny", "source": "random", "split_layer": 2, "hook": "blocks.2.hook_resid_pre",
        "n_layer": 4, "n_head": 4, "d_model": 64, "vocab_size": 50257,
    }))
    argv = ["export_sae.py", "--random", "--model-dir", str(model_dir), "--d-sae", "64"]
    monkeypatch.setattr(sys, "argv", argv)
    assert export_sae.main() == 0  # not checked for random SAEs by default
    assert (model_dir / "sae" / "manifest.json").exists()
    monkeypatch.setattr(sys, "argv", [*argv, "--min-ev", "0.5"])
    assert export_sae.main() == 1
    assert not (model_dir / "sae" / "manifest.json").exists()
    assert "below --min-ev" in capsys.readouterr().err


def test_decoder_norm_rescaling_is_folded_into_the_weights():
    """SAELens TopK with rescale_acts_by_decoder_norm == the plain recipe on the folded weights."""
    w = random_weights()
    x = torch.randn(6, 16)
    s = SaeSettings("topk", 5, center_resid=False, apply_b_dec_to_input=True, prepend_bos=True)
    norms = w["W_dec"].norm(dim=-1)
    # What SAELens computes: scale the pre-activations, top-k, then undo the scale before decoding.
    pre = ((x - w["b_dec"]) @ w["W_enc"] + w["b_enc"]) * norms
    top = pre.topk(5, dim=-1)
    acts = torch.zeros_like(pre).scatter_(-1, top.indices, top.values.relu())
    recon = (acts / norms) @ w["W_dec"] + w["b_dec"]

    folded = sae_format.fold_decoder_norm(w, {"rescale_acts_by_decoder_norm": True})
    assert torch.allclose(sae_format.encode(x, folded, s), acts, atol=1e-5)
    assert torch.allclose(sae_format.decode(sae_format.encode(x, folded, s), folded), recon, atol=1e-5)
    assert torch.allclose(folded["W_dec"].norm(dim=-1), torch.ones(40), atol=1e-6)
    assert sae_format.fold_decoder_norm(w, {}) is w


def test_export_own_sae_from_a_saelens_folder(tmp_path, monkeypatch):
    """--sae-path with a SAELens v6 TopK save: the swap from the open SAE to ours."""
    from safetensors.torch import save_file

    model_dir = tmp_path / "gpt2-random-tiny"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text(json.dumps({
        "name": "gpt2-random-tiny", "source": "random", "split_layer": 2, "hook": "blocks.2.hook_resid_pre",
        "n_layer": 4, "n_head": 4, "d_model": 64, "vocab_size": 50257,
    }))
    run = tmp_path / "prizma-topk-test"
    run.mkdir()
    w = random_weights(d_model=64, d_sae=96)
    save_file(w, run / "sae_weights.safetensors")
    (run / "cfg.json").write_text(json.dumps({
        "architecture": "topk", "k": 8, "d_in": 64, "d_sae": 96, "apply_b_dec_to_input": True,
        "normalize_activations": "none", "rescale_acts_by_decoder_norm": True,
        "metadata": {"prepend_bos": True, "hook_name": "blocks.2.hook_resid_pre",
                     "model_from_pretrained_kwargs": {"center_writing_weights": False}},
    }))
    monkeypatch.setattr(sys, "argv", ["export_sae.py", "--random", "--model-dir", str(model_dir), "--sae-path", str(run)])
    assert export_sae.main() == 0

    manifest, settings = sae_format.read_manifest(model_dir / "sae")
    assert settings == SaeSettings("topk", 8, center_resid=False, apply_b_dec_to_input=True, prepend_bos=True)
    assert manifest["d_sae"] == 96 and manifest["source"] == "prizma-topk-test" and "neuronpedia" not in manifest
    assert manifest["verification"]["mean_l0"] <= 8
    stored = sae_format.read_weights(model_dir / "sae", manifest)
    assert torch.allclose(stored["W_dec"].norm(dim=-1), torch.ones(96), atol=1e-3)  # decoder norms were folded

    # The same SAE as a second folder next to the first (the site shows it with ?sae=sae-own).
    # Labels and concept shards copied there before the export stay, extra keys and all.
    own = model_dir / "sae-own"
    (own / "concepts").mkdir(parents=True)
    (own / "concepts" / "0000.json").write_text("{}")
    (own / "labels.json").write_text(json.dumps({"source": "autointerp", "labels": ["x"] * 96, "scores": [0.9] * 96}))
    first = (model_dir / "sae" / "manifest.json").read_text()
    monkeypatch.setattr(sys, "argv", ["export_sae.py", "--random", "--model-dir", str(model_dir), "--sae-path", str(run),
                                      "--folder", "sae-own"])
    assert export_sae.main() == 0
    assert sae_format.read_manifest(own)[0]["files"]["labels"] == "labels.json"
    assert (own / "concepts" / "0000.json").exists()
    assert (model_dir / "sae" / "manifest.json").read_text() == first
