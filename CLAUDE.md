# Prizma: project rules and working agreement

Prizma runs a small language model (GPT-2 small) entirely in the browser and shows what happens
inside it: which next words it considers, where it attends, which SAE concepts light up, and how
pushing on a concept (steering) changes the text. Full plan: [docs/plan.md](docs/plan.md).
Decisions and their reasons: [docs/decisions.md](docs/decisions.md).

## Hard rules

- **English only.** Code, comments, UI text, docs and commit messages are in English. No i18n layer.
- **Commit and push after every change.** Small, descriptive commits.
- **The site stays fully static.** No accounts, no server-side compute, no models above ~1B params.
- **No new features before Phase 2 ships.** A live MVP comes first; everything else builds on it.
- **`artifacts/` is never committed.** It is the contract between `research/` and `web/` and is
  uploaded to a CDN (Cloudflare R2 or a Hugging Face dataset).

## Architecture

```
research/ (Python, uv)                     artifacts/                      web/ (React + Vite + TS)
  export_onnx.py   GPT-2 -> part A, part B  models/<name>/                   src/engine/  Web Worker, ORT Web,
  export_steering.py steering directions      manifest.json                               SAE encode, steering
  train_sae.py     (Phase 1)                  embeddings.f16.bin             src/views/   Next word, Attention,
  collect_top.py   (Phase 1)                  part_a.onnx, part_b.onnx                    Concepts, Steering
  autointerp.py    (Phase 1)                  steering.json                  src/tours/   guided tours (v2)
  eval.py          (Phase 1)                concepts/ (Phase 1, sharded JSON)
```

The model is split at the residual stream entering block 6 (`blocks.6.hook_resid_pre`):

1. The worker looks up token embeddings in JS from `embeddings.f16.bin` (fp16 Gather is not
   supported on every WebGPU device, and an fp32 table would add ~150 MB).
2. **Part A** (`tok_embeds [1,T,768]` -> `resid [1,T,768]`, `attn [1,6,12,T,T]`) runs blocks 0-5.
3. In between, the browser edits the residual stream: SAE encode for the Concepts view, and
   `resid += strength * typical_norm * direction` for steering.
4. **Part B** (`resid` -> `last_logits [1,V]`, `topk_probs/topk_ids [1,T,10]`, `attn [1,6,12,T,T]`)
   runs blocks 6-11 and the unembedding.

Runtime: ONNX Runtime Web with WebGPU, WASM fallback. Weights are 8-bit block-quantized
(`MatMulNBits`), ~220 MB total. MVP has no KV cache: sequences are capped at 64 tokens and the full
sequence is re-run for every new token.

## Commands

```bash
# Research (Python 3.11+, uv)
cd research
uv sync
uv run --group dev pytest                         # split model must match Hugging Face GPT-2
uv run python export_onnx.py                      # needs huggingface.co; -> artifacts/models/gpt2-l6
uv run python export_steering.py                  # -> artifacts/models/gpt2-l6/steering.json
uv run python export_onnx.py --random --tiny --split-layer 2   # offline test model
uv run python export_steering.py --random --model-dir ../artifacts/models/gpt2-random-tiny

# Web (Node 20+)
cd web
npm install
npm run dev        # serves ../artifacts at /artifacts; open /?model=gpt2-l6
npm test           # unit tests for the engine math
npm run lint
npm run build
```

URL parameters: `model`, `prompt`, `vector`, `strength`, `backend=webgpu|wasm`.

## Hosting

GitHub Pages, deployed by `.github/workflows/pages.yml` on every push to `main` (base path
`/Prizma/`). No custom headers are possible there, so nothing may depend on cross-origin isolation.
Model files come from a CDN set in the `ARTIFACTS_URL` repository variable (e.g.
`https://huggingface.co/datasets/<user>/prizma-artifacts/resolve/main/models`), never from the repo.

## Conventions

- Python: plain PyTorch for the exported graph (`research/prizma_research/gpt2_split.py`), so the
  ONNX graph is small and predictable. Every export verifies itself against Hugging Face and writes
  the numbers into `manifest.json`.
- Web: anything that touches the model runs in `src/engine/model.worker.ts`. Pure math lives in
  small modules (`steering.ts`, `embeddings.ts`) with Vitest tests.
- When the artifact format changes, update `artifacts/README.md`, `protocol.ts` and this file together.

## Roadmap

| Phase | Weeks | Content | Gate to the next phase |
| --- | --- | --- | --- |
| 0 Prototype | 1 | Repo skeleton; GPT-2 split in two and running in the browser on WebGPU; a fixed steering vector from an open SAE changes the output | Steering visibly changes the output in the browser |
| 1 Own SAE | 2 | TopK SAE (16,384 concepts, k=32, layer 6, ~200M OpenWebText tokens, SAELens) on a rented GPU; metrics vs. an open SAE; top examples; Claude Haiku labels + scoring | Metrics close to the open SAE; most of 50 random concepts make sense |
| 2 MVP + launch | 3 | Next word, Attention, Concepts, Steering views and share links; launch on own domain, README, portfolio | Site is live; 5 friends can use it without help |
| 3 Story + promotion | 2 | Concept explorer, 3 guided tours, behind-the-scenes page; blog post, 60 s demo video, Hugging Face release | |
| 4 Second language (optional, parked) | 3 | Out of scope for now: the project is English only | |

## Status

Phase 0 in progress. See the "Status" section of [README.md](README.md).
