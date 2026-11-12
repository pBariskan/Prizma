# Decisions

Short records of choices that are not obvious from the code. Newest first.

## 2026-11-12: Phase 2 views on the open SAE

**The MVP is built on the open GPT-2 SAE first; ours replaces it as an artifact.** Joseph Bloom's
layer 6 residual SAE (`jbloom/GPT2-Small-SAEs-Reformatted`, 24,576 features, Neuronpedia id
`gpt2-small/6-res-jb`) is exported by `research/export_sae.py` into `artifacts/models/<name>/sae/`.
The manifest there carries the whole encode recipe (activation, `k`, centering, `b_dec`, BOS), so
the site does not know which SAE it is running. `export_sae.py --sae-path` converts a SAELens
folder (our Phase 1 TopK SAE) into the same files. Format: `artifacts/README.md`.

**The SAE is encoded in plain JavaScript, from raw fp16 files.** One position costs
24,576 x 768 multiply-adds, about 30 ms in JS, so a whole 64-token run takes a second or two after
the text is finished, in the worker. An ONNX graph would be faster on WebGPU but would bake the
architecture into the graph, need fp32 weights (fp16 shaders are not available everywhere, 8-bit
weights distort small activations) and behave differently per backend. fp16 loses nothing
measurable: explained variance is 0.8726 with the fp32 and with the fp16 weights.

**The residual stream is mean-centered before encoding (`center_resid`).** The open SAE was
trained on a TransformerLens model with `center_writing_weights`, where every residual vector has
zero mean over its 768 dimensions; the Hugging Face weights we export do not. The model's output
is identical (each LayerNorm removes the mean) but the SAE is not: on 169 calibration positions it
explains 87.3% of the variance with centering and 7.2% without. The Hugging Face `cfg.json` of the
SAE does not mention this, so `export_sae.py` measures both conventions on every export and warns
when the configured one is worse. Steering needs no change: a mean component added to the
residual is removed by the next LayerNorm.

**Sequences start with `<|endoftext|>` when the SAE asks for it (`prepend_bos`).** The open SAE
(and SAELens by default) is trained on contexts that begin with BOS, and without it the first
real token becomes GPT-2's attention sink with a huge norm. The engine prepends the token, the UI
shows it as a small "start" marker (attention lines can point at it), and it has no concepts. It
counts towards the 64-token cap.

**Only the encoder is downloaded up front; decoder rows come in shards.** The encoder (37.7 MB
fp16) loads after the model is usable, so Next word, Attention and Steering work first and
Concepts follows. Steering with feature `f` needs one row of `W_dec`: the decoder is split into
96 shards of 256 rows (393 KB each) instead of shipping another 38 MB. Plain files rather than
HTTP range requests, so any static host and the browser cache work.

**Concepts are read from the residual that part B received.** For the steered text that is the
residual after the push, so the steered concept shows up at the top of every token. That is the
honest picture of what the upper half of the model saw, and it makes the mechanism visible.

**SAE features steer in [-1, 1], starting at 0.5.** Strength stays in units of the typical
residual norm (83.6 at layer 6). With the open SAE and "I think that you are": feature 12102
(weddings) at 0.25x barely changes the text, at 0.5x gives "going to be the bride of my daughter",
at 1.0x collapses into "reception reception reception". Feature 9191 (Paris) behaves the same way.
ActAdd vectors keep the [-2, 2] range from `steering.json`; a vector can override both with
`range` and `default_strength`.

**One extra forward pass per run feeds the views.** During generation only the residual and the
top-10 of the last position are read back. When the text is finished, one more pass over the whole
sequence returns the top-10 at every position (so prompt words are inspectable too) and all 144
attention patterns (2.4 MB at 64 tokens). Reading attention back on every step would slow
generation for data nobody sees.

**The engine is one module that runs in the worker and in Node.** `web/src/engine/engine.ts` has
no browser dependencies; `scripts/node-engine.ts` runs it on ONNX Runtime Web's WASM build in
Node. That gives an end-to-end test on the tiny random model (skipped where `artifacts/` does not
exist, such as CI) and `npm run export-demo`, which precomputes `demo.json` with exactly the code
the browser runs.

**A precomputed example is the page's initial state.** While the 220 MB download runs, the site
shows `demo.json` (original and steered text with candidates, attention and concepts, about
300 KB) under the progress bar, and every view works on it. Because it is what the visitor's
browser would compute, nothing changes when the model becomes ready. A share link with its own
text skips the example.

**Downloads are cached under a version.** Artifacts are re-exported under the same file names, so
the Cache API key now includes a hash of the model manifest (or the SAE's `version`); stale copies
of the same file are deleted when a new one is stored.

**Labels come from Neuronpedia's public export for now.** `research/export_labels.py` keeps one
explanation per feature (gpt-4o-mini where available) and writes `sae/labels.json`; 24,564 of
24,576 features have one. "Texts where it fires most" is a link to the feature's Neuronpedia page
until Phase 1 produces our own examples.

**D3 draws the attention lines; React draws everything else.** `d3-selection`, `d3-scale` and
`d3-transition` only (about 30 KB), used for the data join and transitions of the arcs laid over
the text. The default head is layer 4, head 11, a previous-token head whose lines are easy to read.

## 2026-11-07: Hosting on GitHub Pages

The site is deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`
(served from `/Prizma/`, set through `VITE_BASE`). Pages cannot send custom headers, so the site
does not use cross-origin isolation (COOP/COEP); this costs nothing today because WASM already
runs single-threaded. The model files do not go into the repo or the Pages build (Pages has a
100 MB per-file limit and they are research outputs): they are served from a CDN set through the
`ARTIFACTS_URL` repository variable, e.g. a Hugging Face dataset, which sends CORS headers.

## 2026-11-07: Phase 0 prototype

**English only.** All code, UI and docs are in English; there is no i18n layer. The optional
second-language phase from the original plan is parked.

**Split point is `blocks.6.hook_resid_pre`.** The residual stream entering block 6 (output of
blocks 0-5). This matches the hook of Joseph Bloom's open GPT-2 small residual SAEs
(`jbloom/GPT2-Small-SAEs-Reformatted`), which Neuronpedia also indexes, so the open-SAE baseline
and our own Phase 1 SAE can share the same split.

**Own PyTorch forward pass for export.** `research/prizma_research/gpt2_split.py` re-implements
GPT-2 in ~100 lines instead of exporting the Hugging Face modules. Recent `transformers` versions
build the causal mask outside the blocks, which makes calling single blocks error-prone; our version
has an explicit causal mask and returns attention probabilities directly. Tests check it against
Hugging Face to ~1e-6.

**Token embedding lookup happens in JavaScript.** Part A takes token embeddings, not ids. The
table ships as raw fp16 (77 MB). An in-graph fp16 Gather failed on a WebGPU device without
`shader-f16`, and an fp32 table would cost ~150 MB more.

**8-bit weight-only quantization (`MatMulNBits`, block size 32, symmetric).** Supported by both the
WebGPU and WASM backends of ONNX Runtime Web. Total download ~220 MB (embeddings 77 MB, part A
51 MB, part B 91 MB; part B carries the tied unembedding). 4-bit (`--quantize q4`) is available
if size becomes a problem; its quality on real weights is untested.

**Part B returns last-position logits plus top-10 for every position.** Returning full logits for
every position would read back ~13 MB per step at 64 tokens.

**Steering strength is in units of the typical residual norm.** `steering.json` stores unit
directions and the median residual norm at the split layer (ignoring position 0, the attention
sink). Slider value 1.0 adds one typical residual's worth of the direction at every position.

**Phase 0 steering directions.** Activation-addition (ActAdd) vectors computed from contrasting
prompts need nothing but GPT-2 itself; SAE decoder directions are added with
`export_steering.py --sae-feature ID:label` once feature ids are picked on Neuronpedia.

**Single-threaded WASM for now.** Multi-threaded WASM deadlocked when ONNX Runtime is bundled
inside our module worker (its pthread workers re-import the worker script). WebGPU is the main
path; revisit with `ort.env.wasm.wasmPaths` or the non-bundled ORT build.

**Greedy decoding.** Deterministic output makes "before vs. after steering" a fair comparison and
makes share links reproducible.
