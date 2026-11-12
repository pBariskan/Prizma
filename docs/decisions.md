# Decisions

Short records of choices that are not obvious from the code. Newest first.

## 2026-11-12: Phase 0 gate with real weights

Numbers and browser output: [phase0-verification.md](phase0-verification.md).

**The unembedding is not quantized; part B reads it from the embedding table.** The first export
with real weights agreed with Hugging Face on only 91% of top-1 predictions (KL 1.3e-2), and
greedy continuations diverged after a few tokens. Quantizing one group of weights at a time showed
that the unembedding matrix alone causes this: about 60x more KL than all 12 blocks together.
GPT-2 ties that matrix to the token embeddings, which the browser already downloads as fp16, so
`part_b.onnx` now has an external-data initializer pointing at `embeddings.f16.bin` (Cast +
Transpose, folded by ONNX Runtime at session creation) and the worker passes the bytes through
the `externalData` session option. Result: 98.7% top-1 agreement, KL 1.6e-4, and 179 MB instead
of 220 MB. This supersedes "part B carries the tied unembedding" below.

**Asymmetric 8-bit blocks.** One extra byte per 32 weights; KL to Hugging Face drops by a third
compared with symmetric blocks. Block size 16 would cut it by another third for about +13 MB; not
needed now. 4-bit is not usable: 84% top-1 agreement even with an exact unembedding.

**The export checks itself on real sentences and can fail.** The old check used one sequence of
random token ids, where the next-token distribution is nearly flat, and the reported drift (0.04)
came from random weights. The check now uses 16 real prompts: top-1 agreement at every position,
KL, and greedy continuations. `export_onnx.py` exits with an error below `--min-top1` (0.97), so
the publish workflow cannot upload a broken model.

**Every sequence starts with BOS.** The worker prepends `<|endoftext|>` (`manifest.prepend_bos`).
GPT-2 saw it between documents; Joseph Bloom's SAE for this hook was trained with it (and on the
mean-centred residual, which matters for encoding but not for steering); and without it the first
real token becomes the attention sink with a residual norm ~35x the typical one. Steering vectors
and the typical norm (83.1) are computed the same way. The BOS position is steered like every
other position. Leaving it out changes the exact wording in about two thirds of runs (19 of 30),
but not the topic or the strength at which the text collapses, so the simpler rule stays.

**Curated SAE features, each with its own strength range.** `steering.json` ships five features
of the open layer-6 SAE whose Neuronpedia explanation matches what they do when added to the
residual: Weddings (12102), Christmas (6330), Space missions (16285), Cooking and recipes (8888),
God and faith (16860). A single SAE feature is a sharper push than a contrast of two prompts:
clear at 0.5x of the typical norm, repeating one word from 1x. So vectors carry `range` and
`default_strength` ([-1, 1] and 0.5 for SAE features, [-1.5, 1.5] and 0.5 for ActAdd) and the
slider follows the selected vector.

**The ActAdd "weddings" pair is gone.** `" I talk about weddings constantly"` minus
`" I do not talk about weddings"` works in Turner et al. because it is added position by position.
As a mean difference over positions it encodes "do not", and never produced wedding text. Topics
now come from SAE features; ActAdd keeps the two single-word contrasts.

**Model files are cached under their hash.** The Cache API key includes the file's SHA-256 from
`manifest.json`, and older copies of the same file are deleted. Without this, a visitor who had
loaded the site before a re-export would run new graphs against old weights.

**Artifacts are published by a workflow, not from a laptop.** `publish-artifacts.yml` runs the
export on the GitHub runner and uploads `artifacts/models/` to a public Hugging Face dataset with
the `HF_TOKEN` secret. The export is deterministic and self-checking, so nothing has to be copied
between machines and the token never leaves GitHub. Steps: [publishing.md](publishing.md).

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
*Updated 2026-11-12: asymmetric blocks, unembedding no longer quantized or stored in part B,
~180 MB; 4-bit tested and rejected.*

**Part B returns last-position logits plus top-10 for every position.** Returning full logits for
every position would read back ~13 MB per step at 64 tokens.

**Steering strength is in units of the typical residual norm.** `steering.json` stores unit
directions and the median residual norm at the split layer (ignoring position 0, the attention
sink). Slider value 1.0 adds one typical residual's worth of the direction at every position.

**Phase 0 steering directions.** Activation-addition (ActAdd) vectors computed from contrasting
prompts need nothing but GPT-2 itself; SAE decoder directions are added with
`export_steering.py --sae-feature ID:label` once feature ids are picked on Neuronpedia.
*Updated 2026-11-12: five features are picked and exported by default.*

**Single-threaded WASM for now.** Multi-threaded WASM deadlocked when ONNX Runtime is bundled
inside our module worker (its pthread workers re-import the worker script). WebGPU is the main
path; revisit with `ort.env.wasm.wasmPaths` or the non-bundled ORT build.

**Greedy decoding.** Deterministic output makes "before vs. after steering" a fair comparison and
makes share links reproducible.
