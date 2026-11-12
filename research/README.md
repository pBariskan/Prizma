# research/

The offline pipeline that produces everything in `artifacts/`.

| Script | Phase | What it does |
| --- | --- | --- |
| `export_onnx.py` | 0 | Splits GPT-2 at `blocks.{L}.hook_resid_pre` into two ONNX parts, quantizes them, verifies against Hugging Face |
| `export_steering.py` | 0 | Steering directions: ActAdd contrast vectors and open-SAE decoder rows |
| `export_sae.py` | 2 | Converts a SAELens SAE (the open GPT-2 SAE, or ours with `--sae-path`) into the raw files the browser encodes with; verifies explained variance and the residual-centering convention |
| `export_labels.py` | 2 | One label per feature of the open SAE, from Neuronpedia's public export |
| `train_sae.py` | 1 | TopK SAE (16,384 concepts, k=32) on the layer-6 residual stream with SAELens |
| `collect_top.py` | 1 | Top-20 activating snippets per concept over ~10M tokens |
| `autointerp.py` | 1 | Claude Haiku labels and descriptions (Batch API), plus a scoring pass |
| `eval.py` | 1 | Explained variance, L0, LM loss increase, dead-concept rate vs. an open SAE |

Phase 1 scripts do not exist yet.

```bash
uv sync
uv run --group dev pytest
uv run python export_onnx.py --help
```
