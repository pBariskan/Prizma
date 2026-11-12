# Decisions

Short records of choices that are not obvious from the code. Newest first.

## 2026-11-12: Phase 1 pipeline (our own SAE)

**SAELens trains on our own forward pass.** `train_sae.py` hands SAELens a small wrapper
(`HookedGPT2`) around `gpt2_split.GPT2` instead of a TransformerLens model. The SAE therefore
learns on exactly the tensor part A gives the browser: the raw Hugging Face residual stream, with
none of TransformerLens' weight processing. It also keeps us independent of TransformerLens 4,
which replaced `HookedTransformer` with a new bridge class.

**Our SAE reads the raw residual; the open SAE reads a centered one.** Joseph Bloom's `res-jb`
SAEs were trained with `center_writing_weights`, so they expect the residual minus its mean over
the 768 dimensions. `prizma_research/sae.py` records this per SAE (`center_input`). `eval.py`
reports explained variance in both spaces; the centered one is the fair comparison, because the
model itself only ever reads the residual through a LayerNorm, which removes that mean.

**Every context starts with `<|endoftext|>`, and those positions are never used.** The browser
puts BOS in front of every prompt, so training does too. BOS and document-boundary positions are
left out of training, example collection and all metrics: their residual is an outlier (the
attention sink). Training contexts are 128 tokens, examples and evaluation use 64, the longest
sequence the site runs.

**Data.** OpenWebText pre-tokenized for GPT-2 (`apollo-research/Skylion007-openwebtext-tokenizer-gpt2`),
streamed. Training reads from the first shard (200M tokens end in the second of 73); evaluation
reads the last shard and example collection the one before it.

**Training setup.** TopK, 16,384 concepts, k = 32, 200M tokens, batches of 4,096 tokens, Adam at
3e-4 with 1,000 warm-up steps and a linear decay over the last 20%, AuxK loss for dead concepts,
fp32. Activations are scaled to a mean norm of sqrt(768) during training and SAELens folds the
factor into the weights at the end, so the saved SAE needs no normalization at run time.

**A minimal SAE class is the reference for the browser.** `prizma_research/sae.py` is a few lines of
math (`relu` or top-k of `(x - b_dec) @ W_enc + b_enc`). Evaluation and example collection use it
instead of SAELens, every training run ends by comparing the two on fresh activations, and the
SAELens folder (`cfg.json` + `sae_weights.safetensors`) is what `export_sae.py` converts for the site.

**Examples keep token bytes, not token strings.** GPT-2 splits most non-ASCII characters (curly
quotes, dashes, accented letters) over two or three tokens, and a token decoded alone shows as "�".
`collect_top.py` stores the bytes of every token and `examples.merge_utf8` joins the tokens of such
a character when an example is cut out, giving the joined piece the largest activation of its parts.
Found by the first run on real text; 344 of GPT-2's 50,257 tokens hold part of a character.

**Training runs live in `research/runs/`, not in `artifacts/`.** They hold checkpoints, logs,
examples and API state that the site never downloads. Neither folder is committed.

**Labels: Claude Haiku 5.5 through the Batch API.** One request per concept with its 15 strongest
and 5 random firing snippets, firing tokens marked as `<<token>>`; structured output with a label
of at most 6 words and a one-sentence description; "unclear" is an allowed answer and shows as no
label on the site. Low thinking effort, no prompt caching (the shared prefix is small).

**Scores: detection on held-out examples, for every labeled concept.** The model sees only the
label and description plus 5 firing and 5 silent snippets, unmarked and shuffled, and says where
the concept fires. The score is the balanced accuracy (0.5 chance, 1.0 perfect). The firing
snippets are random contexts the labeling pass never saw. Silent snippets are random text, which
makes this a lenient test: it shows whether a label describes what the concept fires on, not
whether it separates it from similar concepts. The plan said to score a sample; scoring all
concepts costs about $2, so we score all.

**Spending money takes an explicit flag.** `autointerp.py` is a dry run unless `--submit` is
given, refuses a pass estimated above `--max-cost`, and writes every batch id to disk before it
waits, so a re-run collects results instead of paying twice. `run_phase1.py` never submits.

**How we read the Phase 1 gate.** "Metrics close to the open SAE" is checked by `eval.py` as:
explained variance (centered) at most 0.05 below the open SAE, LM loss recovered at most 2 points
below it, at most 10% dead concepts. "Most of 50 random concepts make sense" stays a human check
on `runs/<name>/autointerp/review.md`.

**Cost, re-estimated with prices from 2026-11-12.** GPU: one RTX 4090 on RunPod Secure Cloud at
$0.89/h for 2 to 3.5 hours, about $2-4 (the plan said $10-40). Labeling and scoring 16,384
concepts: about $5-6 at $0.05 / $0.25 per million input / output tokens (the plan said $20-50).
Details and the provider comparison are in [gpu-runbook.md](gpu-runbook.md).

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
