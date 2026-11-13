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
  published to a public Hugging Face dataset ([docs/publishing.md](docs/publishing.md)).

## Architecture

```
research/ (Python, uv)                     artifacts/                      web/ (React + Vite + TS)
  export_onnx.py   GPT-2 -> part A, part B  models/<name>/                   src/engine/  Web Worker, ORT Web,
  export_steering.py steering directions      manifest.json                               SAE encode, steering
  export_sae.py    SAE -> web format          embeddings.f16.bin             src/state/   runs, share links, demo
  export_labels.py Neuronpedia labels         part_a.onnx, part_b.onnx       src/views/   Next word, Attention,
  check_steering.py browser pipeline in Py    steering.json, demo.json                    Concepts, Steering
  train_sae.py     (Phase 1)                  sae/ manifest.json, encoder,   scripts/     engine in Node: demo
  collect_top.py   (Phase 1)                       biases, decoder/, labels               export, end-to-end test
  autointerp.py    (Phase 1)                concepts/ (Phase 1, sharded JSON) src/tours/  guided tours (v2)
  eval.py          (Phase 1)
```

The model is split at the residual stream entering block 6 (`blocks.6.hook_resid_pre`):

1. The worker puts BOS (`<|endoftext|>`, `manifest.prepend_bos`) in front of the prompt and looks
   up token embeddings in JS from `embeddings.f16.bin` (fp16 Gather is not supported on every
   WebGPU device, and an fp32 table would add ~150 MB).
2. **Part A** (`tok_embeds [1,T,768]` -> `resid [1,T,768]`, `attn [1,6,12,T,T]`) runs blocks 0-5.
3. In between, the browser edits the residual stream: `resid += strength * typical_norm * direction`
   for steering, then SAE encode (plain JS, `sae.ts`) for the Concepts view. How to encode is
   written in `sae/manifest.json`, so swapping the SAE is an artifact change only. The open GPT-2
   SAE needs two things that are easy to miss: the residual is mean-centered first (`center_resid`)
   and every sequence starts with `<|endoftext|>` (`prepend_bos`, in the model or the SAE manifest).
4. **Part B** (`resid` -> `last_logits [1,V]`, `topk_probs/topk_ids [1,T,10]`, `attn [1,6,12,T,T]`)
   runs blocks 6-11 and the unembedding. GPT-2 ties the unembedding to the token embeddings, so
   `part_b.onnx` does not contain it: the graph points at `embeddings.f16.bin` as ONNX external
   data (`manifest.external_data`) and the worker passes the bytes it already downloaded.

Runtime: ONNX Runtime Web with WebGPU, WASM fallback (two builds of the runtime, each downloaded
only when it is used). The transformer blocks are 8-bit block-quantized (`MatMulNBits`,
asymmetric); the unembedding is never quantized (it is the one matrix where 8 bits visibly change
the predictions). ~180 MB total, plus 38 MB for the SAE encoder (downloaded after the model is
usable). MVP has no KV cache: sequences are capped at 64 tokens (BOS included) and the full
sequence is re-run for every new token. When a run finishes, one more pass returns the candidates
at every position and all attention patterns.

## Commands

```bash
# Research (Python 3.11+, uv)
cd research
uv sync
uv run --group dev pytest                         # split model must match Hugging Face GPT-2
uv run python export_onnx.py                      # needs huggingface.co; -> artifacts/models/gpt2-l6
uv run python export_steering.py                  # -> artifacts/models/gpt2-l6/steering.json
uv run python export_sae.py                       # open SAE -> artifacts/models/gpt2-l6/sae
uv run python export_sae.py --sae-path runs/<name>   # our own SAE (SAELens folder) instead
uv run python export_labels.py                    # Neuronpedia labels -> .../sae/labels.json
uv run python check_steering.py                   # original vs. steered text, same pipeline as the browser
uv run python export_onnx.py --random --tiny --split-layer 2   # offline test model
uv run python export_steering.py --random --model-dir ../artifacts/models/gpt2-random-tiny
uv run python export_sae.py --random --model-dir ../artifacts/models/gpt2-random-tiny

# Web (Node 20+)
cd web
npm install
npm run dev        # serves ../artifacts at /artifacts; open /?model=gpt2-l6
npm test           # unit tests; plus the engine end to end if artifacts/models/gpt2-random-tiny exists
npm run lint
npm run build
npm run export-demo -- --vector sae-12102 --strength 0.5   # -> artifacts/models/gpt2-l6/demo.json
```

URL parameters. Share link: `prompt`, `new` (tokens to generate), `vector` (a `steering.json` id or
`sae-<feature id>`), `strength`, `token` (`o7` / `s7`: token 7 of the original / steered text),
`view=next|attention|concepts|steer`, `head=<layer>.<head>`, `concept` (pinned feature id).
Development: `model`, `sae` (another SAE folder of the model), `backend=webgpu|wasm`.

## Hosting

GitHub Pages, deployed by `.github/workflows/pages.yml` on every push to `main` (base path
`/Prizma/`). No custom headers are possible there, so nothing may depend on cross-origin isolation.
Model files come from a public Hugging Face dataset, never from the repo. Its address
(`https://huggingface.co/datasets/poyri/prizma-artifacts/resolve/main/models`) is the default
in `pages.yml`; the `ARTIFACTS_URL` repository variable replaces it when set.
`.github/workflows/publish-artifacts.yml` (run by hand) exports with real weights on the runner and
uploads there, using the `HF_TOKEN` repository secret. Steps and links: [docs/publishing.md](docs/publishing.md).
The browser caches model files under their SHA-256 from `manifest.json`, so a re-export needs no
cache busting.

## Conventions

- Python: plain PyTorch for the exported graph (`research/prizma_research/gpt2_split.py`), so the
  ONNX graph is small and predictable. Every export verifies itself against Hugging Face on real
  sentences (not random token ids, and not only random weights: both hid a real problem once),
  writes the numbers into `manifest.json` and fails below `--min-top1`.
- Steering strengths are in units of the typical residual norm. SAE features work at about
  0.25x-0.75x, ActAdd vectors up to about 1x; each vector in `steering.json` carries its own
  `range` and `default_strength`.
- Web: anything that touches the model runs in the worker. The pipeline itself is
  `src/engine/engine.ts`, which has no browser dependencies: `model.worker.ts` hosts it in the
  browser and `scripts/node-engine.ts` in Node (demo export, end-to-end test). Pure math lives in
  small modules (`steering.ts`, `embeddings.ts`, `sae.ts`) with Vitest tests.
- Safari gives a Web Worker about 512 KB of stack, and its engine spends 8 bytes of it on every entry
  of an array literal that holds anything but constants. Nothing large goes into the worker as a
  literal: the tokenizer's vocabulary is rewritten to JSON text at build time
  (`scripts/vocabulary-as-json.ts`, in `plugins` and in `worker.plugins`). The cloud machine runs
  Chromium only, so a browser check from there is a Chromium check and is reported as one.
- The UI keeps what is on screen in the URL (`src/state/share.ts`); a new piece of view state
  that should survive a share link goes there.
- The page is one sheet with four views, one on screen at a time. Each view has one colour
  (`--next`, `--attention`, `--concepts`, `--steer` in `src/index.css`); inside the sheet `--lens` is
  the colour of the open view, and nothing else on the page is coloured. Fonts are bundled
  (`@fontsource-variable/*`), never loaded from another host.
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

Phase 0 gate passed on 2026-11-12 with real GPT-2 weights
([docs/phase0-verification.md](docs/phase0-verification.md)). The repository is public and the
site is live at <https://pbariskan.github.io/Prizma/> since the same day, with the Phase 2 views on
the open GPT-2 SAE. See the status line at the top of [README.md](README.md).
