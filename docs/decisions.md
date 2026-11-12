# Decisions

Short records of choices that are not obvious from the code. Newest first.

## 2026-11-12: Public repository, live site, dataset address in the workflow

Poyri made the repository public and switched GitHub Pages on (source: GitHub Actions). The site
is at <https://pbariskan.github.io/Prizma/>; the model files are in the public dataset
<https://huggingface.co/datasets/poyri/prizma-artifacts>, exported and uploaded by the
*Publish model artifacts* workflow on a GitHub runner. Before the switch every branch and commit
was searched for tokens, keys and credential files; there were none.

**The dataset address is the default in `pages.yml`; the `ARTIFACTS_URL` variable only replaces
it.** The address used to exist only as a repository variable. It is public, it changes only when
the files move to another dataset, and a variable is a setting that nobody sees in the code: a
fork, or a repository whose variable was deleted, built a site that could not load its model.
Now a build with no settings works, and a fork that publishes its own files sets the variable.

## 2026-11-12: One view at a time (page redesign)

Poyri's feedback on the first Phase 2 page: it was a stack of boxes, it showed everything at once,
and it had no character. The page was rebuilt around these choices.

**One sheet, four views, one on screen at a time.** The page is a header, a short introduction and
a single sheet that holds the prompt, four tabs and the open view. The tabs are numbered in the
order a first visit should take and each asks one question: Next word ("What else could it have
said?"), Attention ("Where was it looking?"), Concepts ("What did it have in mind?"), Steering
("What if you push on it?"). Steering used to be a permanent panel above two permanent texts; it
is now the fourth view (`view=steer` in share links), so a first-time visitor meets one idea at a
time.

**The views that look inside show one text with its details beside it.** Next word, Attention and
Concepts show the original or the steered text (a switch appears once there is a steered one) and,
next to it, what the view knows about one word. The details used to sit below both texts, a scroll
away from the word they describe. Only the Steering view shows both texts side by side, because
only there the comparison is the point; a click on a word there opens it in the view that was used
last.

**Pointing previews, clicking keeps.** While the pointer is over the text the details follow the
word under it, and they return to the clicked word when the pointer leaves the text (not when it
leaves a word, or the gaps between lines would flicker). The floating candidate list on hover
remains only in the Steering view, where there is no detail column.

**A concept's actions appear when it is picked.** The concept list had 20 buttons (Steer and
Examples on each of 10 rows). A row is now a name, a value and a bar; clicking it highlights the
concept across the text and shows "Steer with it" and "Examples" for that row only. Steering from
there opens the Steering view.

**The identity is a prism.** One text goes in, four views come out. Each view has one colour
(orange, teal, violet, magenta, in tab order) and only the open view shows its colour on the text
and in its charts (`--lens` in `index.css`); everything else is ink on paper. The model's words are
set in a serif (Fraunces), the interface in Inter, tokens and numbers in JetBrains Mono.

**Fonts are bundled, not loaded from a font service.** They come from `@fontsource-variable/*`
(SIL Open Font License) and are served with the site, about 0.3 MB for Latin text. A request to a
third-party font host on every visit would sit badly with "nothing leaves your browser", and the
site stays one static bundle.

**Three example starts are offered under the prompt.** Many short prompts loop under greedy
decoding ("I love the cold, but I also love the cold"), which is a poor first impression. The
three starts were picked from ten tried with the real model because the plain continuation and the
Weddings and Space missions continuations all stay readable.

**Enter runs the model.** The prompt is a single growing line, so Enter generates and Shift+Enter
breaks the line (it used to be Ctrl/Cmd+Enter).

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

**Sequences start with `<|endoftext|>` when the model or the SAE asks for it (`prepend_bos`).**
The open SAE (and SAELens by default) is trained on contexts that begin with BOS, and without it
the first real token becomes GPT-2's attention sink with a huge norm. The model manifest is the
main switch, because `steering.json` is computed on the same kind of sequence; the SAE manifest
is honoured as well, so an SAE that needs BOS still gets it on a model exported without the
field. The engine prepends the token, the UI shows it as a small "start" marker (attention lines
can point at it), and it has no concepts. It counts towards the 64-token cap.

**Only the encoder is downloaded up front; decoder rows come in shards.** The encoder (37.7 MB
fp16) loads after the model is usable, so Next word, Attention and Steering work first and
Concepts follows. Steering with feature `f` needs one row of `W_dec`: the decoder is split into
96 shards of 256 rows (393 KB each) instead of shipping another 38 MB. Plain files rather than
HTTP range requests, so any static host and the browser cache work.

**Concepts are read from the residual that part B received.** For the steered text that is the
residual after the push, so the steered concept shows up at the top of every token. That is the
honest picture of what the upper half of the model saw, and it makes the mechanism visible.

**SAE features steer in [-1, 1], starting at 0.5.** Strength stays in units of the typical
residual norm (83.1 at layer 6, measured with BOS). With the open SAE and "I think that you are":
feature 12102 (weddings) at 0.25x barely changes the text, at 0.5x gives "going to be the bride
of a big day", at 1.0x collapses into "reception reception reception". Feature 9191 (Paris)
behaves the same way. The curated directions in `steering.json` carry their own `range` and
`default_strength`; the [-1, 1] and 0.5 in the site are the fallback for a feature picked in the
Concepts view, and `suggested_range` is the fallback for anything else.

**The example prompt is "I have some big news:" with the Weddings feature at 0.5x.** Greedy GPT-2
loops on many short prompts ("I think that you are right. I think that you are right."), which
makes a poor first impression. This one continues with a book announcement and the steered text
with "I'm going to be the bride of my first ever wedding", so the effect is obvious at a glance.

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

**A precomputed example is the page's initial state.** While the model download runs (about 180 MB), the site
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

**Neuronpedia's concept labels are published with credit (Poyri, 2026-11-12).** `sae/labels.json`
holds the explanations Neuronpedia publishes for the open SAE, and the site needs them until our
own SAE and labels exist. The license of that export could not be confirmed, so the dataset card
credits Neuronpedia and says the labels are not covered by the card's MIT tag. The alternative,
unlabelled concepts with a link to Neuronpedia, was rejected because the Concepts view is hard to
read without names.

**D3 draws the attention lines; React draws everything else.** `d3-selection`, `d3-scale` and
`d3-transition` only (about 30 KB), used for the data join and transitions of the arcs laid over
the text. The default head is layer 4, head 11, a previous-token head whose lines are easy to read.

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
