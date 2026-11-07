# Decisions

Short records of choices that are not obvious from the code. Newest first.

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
