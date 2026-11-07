# Prizma: project plan

Condensed English version of the original plan (November 7, 2026).

## Summary

Prizma is an interactive website that runs a small language model entirely in the browser and
shows what is happening inside it. You type a sentence and see which words the model considered,
where it looked, and which concepts lit up inside it. You can then turn those concepts up or down
with a slider and watch the model's behaviour change.

- **Goal:** a portfolio piece, not revenue. Someone looking at it should conclude "this student
  really understands what goes on inside these models".
- **Audience:** recruiters and engineers at AI companies (technical depth), professors and
  researchers (the behind-the-scenes page), and the curious public (guided tours, shareable moments).
- **What is different:** similar projects either show the architecture (Transformer Explainer,
  bbycroft's LLM Visualization) or offer a concept database (Neuronpedia). Prizma combines a live
  model, its own trained concepts and steering in one browser experience.
- **Name:** a prism splits light into its colours; Prizma splits one word of the model into the
  concepts inside it.

## Product

One main screen: a text box with four views into the model underneath.

| Feature | What the user sees | Release |
| --- | --- | --- |
| Next word | Hovering a generated word shows the model's top 10 alternatives and their probabilities | MVP |
| Attention | Clicking a word draws lines to the earlier words it attended to; layer and head selector | MVP |
| Concepts | The 10 strongest SAE concepts on the selected word, with human-readable labels; clicking one shows the texts where it fires most | MVP |
| Steering | Dragging a concept's slider regenerates the text; before and after side by side | MVP |
| Share link | Sentence, selected concept and slider values are stored in the URL | MVP |
| Concept explorer | Search all concepts, browse by category | v2 |
| Guided tours | Three scroll-driven stories for non-technical visitors | v2 |
| Behind the scenes | How the SAE was trained: metrics, charts, failures, lessons | v2 |

Out of scope: user accounts, server-side compute, large models (1B+). The site stays fully static.
The project is English only.

## Technical architecture

Two parts: a Python research pipeline that runs once (SAE training, labelling, export) and a static
website that consumes its outputs. All computation happens in the visitor's browser.

### Model

- GPT-2 small (124M parameters): the most studied model in interpretability, with open SAEs to
  compare against.
- Shipped to the browser quantized (~125-250 MB), downloaded on first visit and then cached.

### Running in the browser

- The model is split at layer 6 into two ONNX parts. Part A goes from the input to that layer's
  residual stream and also outputs attention. Part B goes from there to the output probabilities.
- In between, the browser runs the SAE: it decomposes the residual into concepts (Concepts view)
  and, for steering, adds the chosen concept's direction times the slider value (Steering view).
- ONNX Runtime Web on WebGPU, with a slower WASM fallback.
- No KV cache in the MVP: texts are capped at 64 tokens and the sequence is re-run per token.
- Optional later depth: hand-written WebGPU shaders for the forward pass.

### SAE training (Phase 1)

- TopK SAE, 16,384 concepts, k = 32, on GPT-2's layer-6 residual stream.
- ~200M tokens of OpenWebText, SAELens, one rented A100/H100 for a few hours.
- Metrics: explained variance, mean L0, LM loss increase with the SAE spliced in, dead-concept
  rate; all compared against an open GPT-2 SAE.

### Concept labels and examples (Phase 1)

- For each concept, the top 20 activating snippets over ~10M tokens.
- Claude Haiku writes a short label and a one-sentence description per concept (Batch API).
- A sample of labels is scored in a second pass: Haiku sees only the label and predicts where the
  concept fires. The score goes on the behind-the-scenes page.
- Outputs are sharded JSON on a CDN; the site downloads only the shard for the concept it opens.

### Frontend

- React + Vite + TypeScript, static build. The model runs in a Web Worker.
- D3 for attention lines and charts. Hosting on GitHub Pages (decided 2026-11-07; the plan originally said Vercel or Cloudflare Pages).

## Roadmap

About 8 weeks at 10-15 hours a week; the site goes live around week 6.

| Phase | Weeks | Content | Gate |
| --- | --- | --- | --- |
| 0 Prototype | 1 | Repo skeleton; GPT-2 split in two, running in the browser on WebGPU; a fixed steering vector from an open SAE changes the output | Steering visibly changes the output in the browser |
| 1 Own SAE | 2 | TopK SAE on a rented GPU, metrics vs. an open SAE; example collection; Claude Haiku labels and scores for 16k concepts | Metrics close to the open SAE; most of 50 random concepts make sense |
| 2 MVP + launch | 3 | Next word, Attention, Concepts, Steering views and share link; launch on own domain, README, portfolio | Site live; 5 friends can use it without help |
| 3 Story + promotion | 2 | Concept explorer, 3 guided tours, behind-the-scenes page; blog post, 60 s demo video, Hugging Face release | |

Most important rule: no new features until Phase 2 ships a live version.

## Cost (estimates, check current prices before spending)

| Item | Estimate |
| --- | --- |
| Rented GPU (SAE training + example collection, a few attempts) | $10-40 |
| Claude Haiku labelling + scoring of 16k concepts (Batch API) | $20-50 |
| Hosting (GitHub Pages + a Hugging Face dataset for model files) | free |
| Domain (optional) | $10-15 / year |

## Risks

| Risk | What happens | Mitigation |
| --- | --- | --- |
| Download size | 125-250 MB slows the first visit; people leave | Precomputed demo sentence while loading, progress bar, int8 weights |
| WebGPU support | Some browsers (notably Safari) lack full WebGPU | WASM fallback and a "best in Chrome" note |
| SAE quality | A weak SAE makes concepts meaningless | Build the MVP on an open GPT-2 SAE first, swap ours in later |
| Steering breaks text | High slider values produce gibberish | Safe range per concept, curated working examples |
| GPT-2 writes poorly | Users judge model quality | Good demo sentences; say clearly the point is the internals, not quality |
| Scope creep | The project never ships | Live release at the end of Phase 2, add on top of it |

## Deliverables

1. Live site on its own domain, featured on the portfolio.
2. Public GitHub repo with a good README, architecture diagram and setup steps.
3. SAE weights and concept dataset on Hugging Face, with a model card and metrics.
4. Behind-the-scenes write-up (blog or on the site), possibly for LessWrong / Alignment Forum or an
   interpretability workshop.
5. A 60-second demo video of the steering slider changing the text.
