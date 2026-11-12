"""The export scripts run end to end on a tiny random model and write the contract the web engine reads."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx

from check_steering import SplitModel

RESEARCH = Path(__file__).resolve().parent.parent
FILES = {"embeddings.f16.bin", "part_a.onnx", "part_b.onnx"}


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=RESEARCH, check=True, capture_output=True)


def test_export_and_steering_on_a_tiny_random_model(tmp_path):
    run("export_onnx.py", "--random", "--tiny", "--split-layer", "2", "--out", str(tmp_path))
    model_dir = tmp_path / "gpt2-random-tiny"
    manifest = json.loads((model_dir / "manifest.json").read_text())

    assert manifest["prepend_bos"] is True and manifest["bos_token_id"] == 50256
    assert manifest["external_data"] == {"part_b": ["embeddings.f16.bin"]}
    assert set(manifest["bytes"]) == set(manifest["sha256"]) == FILES
    for report in manifest["verification"].values():
        assert report["top1_agreement"] > 0.9 and report["kl_mean"] < 1e-3

    # Part B takes its unembedding from the embedding table instead of carrying a copy
    # (an fp32 copy alone would be 12.9 MB for this model).
    part_b = onnx.load(str(model_dir / "part_b.onnx"), load_external_data=False)
    external = [i for i in part_b.graph.initializer if i.data_location == onnx.TensorProto.EXTERNAL]
    assert [i.name for i in external] == ["wte_f16"]
    assert {e.key: e.value for e in external[0].external_data}["location"] == "embeddings.f16.bin"
    assert manifest["bytes"]["part_b.onnx"] < 1_000_000

    run("export_steering.py", "--random", "--model-dir", str(model_dir))
    steering = json.loads((model_dir / "steering.json").read_text())
    assert steering["vectors"], "no steering vectors written"
    for v in steering["vectors"]:
        assert abs(np.linalg.norm(v["vector"]) - 1) < 1e-4
        lo, hi = v["range"]
        assert lo < 0 < v["default_strength"] <= hi

    # The Python copy of the browser pipeline runs on the result, and steering changes the output.
    model = SplitModel(model_dir)
    ids = [manifest["bos_token_id"], 10, 200, 3000]
    plain = model.generate(ids, 6)
    assert len(plain) == 6 and plain == model.generate(ids, 6)
    assert model.generate(ids, 6, steering["vectors"][0]["id"], 1.5) != plain
