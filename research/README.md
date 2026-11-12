# research/

The offline pipeline that produces everything in `artifacts/`.

| Script | Phase | What it does |
| --- | --- | --- |
| `export_onnx.py` | 0 | Splits GPT-2 at `blocks.{L}.hook_resid_pre` into two ONNX parts, quantizes them, verifies against Hugging Face |
| `export_steering.py` | 0 | Steering directions: ActAdd contrast vectors and open-SAE decoder rows |
| `run_phase1.py` | 1 | All of Phase 1 in one command: smoke test, the four scripts below in order, skips what is done |
| `train_sae.py` | 1 | TopK SAE (16,384 concepts, k=32) on the layer-6 residual stream with SAELens |
| `collect_top.py` | 1 | Per concept: the 20 strongest and 10 random firing contexts over ~10M held-out tokens |
| `eval.py` | 1 | Explained variance, L0, LM loss with the SAE spliced in, dead concepts; next to the open SAE |
| `autointerp.py` | 1 | Claude Haiku labels and descriptions (Batch API), a scoring pass, and the export for the site |

```bash
uv sync
uv run --group dev pytest
uv run python export_onnx.py --help
```

## Phase 1: our own SAE

```bash
uv run python run_phase1.py --random --tiny    # the whole pipeline offline on a CPU, about half a minute
uv run python run_phase1.py --tiny             # real GPT-2 and data, small budgets (needs huggingface.co)
uv run python run_phase1.py                    # the real run on a rented GPU: see ../docs/gpu-runbook.md
uv run python autointerp.py --run runs/<name>  # labeling dry run: prints the cost, sends nothing
```

Every script takes `--random` (random GPT-2 weights, synthetic text, no downloads) and `--tiny`
(small budgets); together they use the same 4-layer test model as `export_onnx.py --random --tiny`.
`autointerp.py` has `--fake` instead: an offline stand-in for the API.

Two steps cost money and are only ever started by a person: renting the GPU, and
`autointerp.py --submit` (needs `ANTHROPIC_API_KEY`).

A run is a folder in `runs/` (not committed):

```
runs/gpt2-l6-topk16384-k32/
  cfg.json, sae_weights.safetensors   the SAE in SAELens format; export_sae.py turns it into the site artifact
  train_log.jsonl, train_summary.json loss, explained variance, L0, dead concepts, speed
  checkpoints/                        SAELens checkpoints; `train_sae.py --resume-from auto` continues from the newest
  examples/                           examples.safetensors, vocab.json, meta.json (collect_top.py)
  eval.json, eval.md                  metrics next to the open SAE and the Phase 1 gate check (eval.py)
  autointerp/                         state.json (batch ids, usage), labels.jsonl, scores.jsonl, review.md, summary.json
  web/sae/                            labels.json and concepts/NNNN.json for the site (autointerp.py)
```

`web/sae/labels.json` is `{"source": ..., "labels": [string | null], "scores": [number | null]}`,
indexed by concept id. `web/sae/concepts/NNNN.json` holds concepts `[NNNN * 32, (NNNN + 1) * 32)`:

```json
{"concepts": [{"id": 0, "label": "sports scores", "description": "Fires on ...", "score": 0.9,
               "max_act": 41.2, "density": 0.0004,
               "examples": [{"tokens": [" won", " 3", "-", "1"], "acts": [0, 12.5, 3.1, 14.0]}]}]}
```

Code map: `prizma_research/lm.py` (model loading, the residual at the split), `data.py` (token
streams), `sae.py` (the minimal SAE the browser math follows), `examples.py`, `metrics.py`,
`interp.py` (prompts, scoring, Batch API client).
