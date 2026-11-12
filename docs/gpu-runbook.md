# GPU runbook: training Prizma's own SAE (Phase 1)

One command on a rented GPU trains the SAE, collects examples and evaluates it against the open
SAE. A second command, which you start yourself, labels the concepts with Claude Haiku.

**Nothing here has been started.** Both paid steps (renting the GPU, sending the labeling batches)
are yours to start. Prices were checked on 2026-11-12; check them again before you pay.

| | Recommendation | Expected |
| --- | --- | --- |
| Provider | RunPod, Secure Cloud | $10 minimum top-up |
| GPU | 1x RTX 4090 (24 GB) at $0.89/h | 2 to 3.5 hours, **about $2 to $4** |
| Labeling | Claude Haiku 5.5, Batch API | **about $5 to $6** for all 16,384 concepts |

The original plan budgeted $10-40 for the GPU and $20-50 for labeling. Both are lower now: the
job is small for a modern GPU, and Claude Haiku 5.5 through the Batch API costs $0.05 per million
input tokens and $0.25 per million output tokens.

What is not verified: the development environment cannot reach huggingface.co, so the pipeline
has only ever run on a random-weight model with synthetic text (`--random --tiny`, 36 tests). Real
GPT-2, OpenWebText and the open SAE are first touched by the smoke test in step 2, which takes a
few minutes and fails early if something is wrong. The time estimates below are calculated, not
measured; the training log prints the measured speed and the minutes left after its first 100 steps.

## 1. Which provider and GPU

The job needs under 8 GB of GPU memory (GPT-2 small is 124M parameters, the SAE 25M), so any
24 GB card is plenty. A bigger GPU does not pay off here: the run is short and limited as much by
the SAE's small batches as by raw compute.

Prices per GPU-hour on 2026-11-12. RunPod prices are from its pricing page ("updated November 9,
2026"); Lambda and Vast.ai numbers are from third-party trackers because Lambda shows current
prices only after login and Vast.ai is a marketplace whose prices move by the hour.

| Provider | RTX 3090 | RTX 4090 | RTX 5090 | A100 80 GB | H100 | Billing |
| --- | --- | --- | --- | --- | --- | --- |
| RunPod Secure Cloud | $0.50 | **$0.89** | $1.19 | $1.79 | $2.89 | per second, $10 minimum top-up, no egress fees |
| RunPod Community Cloud | $0.22 | $0.34 | $0.69 | from $1.19 | from $1.99 | same; hosts are third parties, availability varies |
| Vast.ai (marketplace) | ~$0.17 | ~$0.34-0.60 | ~$0.49 | ~$0.94-1.00 | ~$2.14 | per second, $5 minimum deposit, storage and bandwidth billed separately, host quality varies |
| Lambda | not offered | not offered | not offered | A100 40 GB $1.99 | $3.29 ($3.79 from Nov 16) | per minute, card only, often out of stock; cheapest in stock was an A10 24 GB at $1.29 |

Why RunPod Secure Cloud with an RTX 4090:

- The whole run costs a few dollars anywhere, so reliability matters more than the hourly price.
  Secure Cloud runs in data centers with fast downloads; a Community or Vast.ai host that is slow
  or disappears costs more time than the $1-2 it saves.
- Per-second billing and no egress fees: you pay for the minutes used and downloading the results
  is free.
- If no RTX 4090 is free: RTX 5090 (faster, $1.19), RTX 3090 or RTX A5000 (slower, cheaper), or
  the same cards on Community Cloud. Any NVIDIA card with 16 GB or more works unchanged.

Time and cost on one RTX 4090 at $0.89/h:

| Step | Tokens | Time (estimate) |
| --- | --- | --- |
| Setup: clone, install, download GPT-2 | | 5-10 min |
| Smoke test (all four steps, small) | 1.5M | ~5 min |
| Training | 200M | 1-2 h |
| Example collection | 10M | ~5 min |
| Evaluation, ours and the open SAE | 4M | ~5 min |
| Waiting for the labeling batches (optional, can also run from your Mac) | | up to ~1-2 h |
| **Total** | | **2 to 3.5 h, about $2 to $4** |

## 2. Run it

You need a RunPod account with $10 of credit. A Hugging Face token is optional (everything
downloaded is public; a token only avoids rate limits and is needed for the upload in step 5).

1. On RunPod, deploy a pod: **Secure Cloud, 1x RTX 4090, On-Demand** (not Spot), a recent
   "RunPod PyTorch" template, **30 GB container disk and 30 GB volume**. Open its web terminal or
   connect over SSH.
2. Paste this one command:

   ```bash
   git clone https://github.com/pBariskan/Prizma.git /workspace/Prizma && bash /workspace/Prizma/research/gpu_run.sh
   ```

   Until the Phase 1 pull request is merged, add `--branch claude/project-thread-v2tlr7` after
   `git clone`. If the repository is private, GitHub asks for a user name and a personal access
   token.

`gpu_run.sh` installs `uv` and the environment, checks that PyTorch sees the GPU, starts
`run_phase1.py` in the background and follows `research/phase1.log`. Closing the terminal does not
stop the run; the same command shows the log again.

What `run_phase1.py` does, in order:

| Step | Script | Output in `research/runs/gpt2-l6-topk16384-k32/` |
| --- | --- | --- |
| Smoke test | all four below with `--tiny`, into `runs/smoke/` | proves that model, dataset, open SAE and GPU work |
| Train | `train_sae.py` | `cfg.json`, `sae_weights.safetensors` (SAELens format), `train_log.jsonl`, `train_summary.json` |
| Examples | `collect_top.py` | `examples/`: 20 strongest and 10 random contexts per concept |
| Evaluate | `eval.py` | `eval.json`, `eval.md`: metrics next to the open SAE, gate check |
| Labeling dry run | `autointerp.py` | prints what labeling will cost; sends nothing |

While it runs, the training log looks like this, one line every 100 steps:

```
[train] step   4000     16.38M tok  mse ...  EV 0.85  L0 32.0  dead 12  61.3k tok/s  50 min left
```

`EV` (explained variance) should climb quickly and flatten, `L0` stays at 32, and `dead` should
stay a small share of the 16,384 concepts.

## 3. Read the result

The run ends by printing `eval.md`: our SAE next to Joseph Bloom's open `res-jb` SAE for the same
layer (24,576 concepts, the SAE the MVP uses today).

The Phase 1 gate has two parts. The first is checked by `eval.py` and reads "metrics close to the
open SAE" as: explained variance at most 0.05 below the open SAE, LM loss recovered at most 2
points below it, and at most 10% dead concepts. The thresholds are in
`research/prizma_research/metrics.py`; they are our reading of "close", so judge the table yourself
too. Our SAE has fewer concepts and a lower L0 (32 against about 50), so slightly lower explained
variance is expected.

If the gate fails, the cheapest things to try are a longer run (`--training-tokens 400000000`)
and a higher `k`; both are flags of `train_sae.py`.

## 4. Label the concepts (paid, you start it)

This needs an Anthropic API key with a few dollars of credit. It can run on the pod while it is
still up, or on your Mac after step 5.

```bash
cd /workspace/Prizma/research
export ANTHROPIC_API_KEY=...      # from console.anthropic.com
RUN=runs/gpt2-l6-topk16384-k32

uv run python autointerp.py --run $RUN                         # dry run: prints the cost again, sends nothing
uv run python autointerp.py --run $RUN --sample 200 --submit   # pilot on 200 random concepts: about $0.07
less $RUN/autointerp/review.md                                 # read the labels next to their examples
uv run python autointerp.py --run $RUN --submit                # all concepts: about $5 to $6
```

- Each `--submit` call prints the estimate first and refuses to send a pass estimated above
  `--max-cost` (default $15). Add `--count-tokens` to the dry run for an exact input-token count
  from the API (free).
- Batches usually finish within an hour. The command waits and can be interrupted: running it
  again collects the results and never sends a concept twice (`autointerp/state.json` holds the
  batch ids). `--no-wait` submits and exits.
- After the pilot, the summary line shows the real token usage. Multiply by 82 for the full run
  before you start it.
- The second gate, "most of 50 random concepts make sense", is yours to judge: `review.md` lists
  50 random labeled concepts with their label, score and strongest examples.

How the estimate is made: one labeling request per concept is about 2,000 input tokens (the
instructions plus 20 snippets) and one scoring request about 950, with 300-400 output tokens each
including the model's short thinking. That is roughly 33M + 15M input tokens and 11M output tokens:
$1.7 + $0.8 + $2.9 = about $5.3 through the Batch API (twice that without it). The input side is
counted from the real prompts with a conservative characters-per-token ratio; the output side is
the uncertain part, and the pilot measures it.

## 5. Save the results and stop paying

Everything is in `research/runs/gpt2-l6-topk16384-k32/` (about 350 MB without `checkpoints/`).
The simplest way off the pod is a private Hugging Face model repository:

```bash
cd /workspace/Prizma/research
uv run hf auth login                       # a token with write access
uv run hf upload <your-hf-user>/prizma-sae-gpt2-l6 runs/gpt2-l6-topk16384-k32 . --private --exclude "checkpoints/*"
```

Then **terminate the pod** on RunPod (a stopped pod still bills its volume). To continue on your
Mac: `uv run hf download <your-hf-user>/prizma-sae-gpt2-l6 --local-dir research/runs/gpt2-l6-topk16384-k32`.

## 6. Put it on the site

The site reads the SAE from `artifacts/models/gpt2-l6/sae/`. With the Phase 2 export script
(`research/export_sae.py`, on the Phase 2 branch until it merges):

```bash
cd research
uv run python export_sae.py --sae-path runs/gpt2-l6-topk16384-k32     # weights -> artifacts/models/gpt2-l6/sae/
cp -r runs/gpt2-l6-topk16384-k32/web/sae/. ../artifacts/models/gpt2-l6/sae/   # labels.json, concepts/
```

Then upload `artifacts/` to the CDN as for Phase 0. No web code changes.

## If something goes wrong

| What you see | What to do |
| --- | --- |
| The smoke test fails while downloading | Usually a Hugging Face rate limit or outage. `export HF_TOKEN=...` (a free read token) and run the same command again. |
| `PyTorch does not see the GPU` | The pod has no GPU or an old driver. Deploy a new pod from a recent RunPod PyTorch template. |
| The pod stopped or training crashed | Run the same command again. Finished steps are skipped and training continues from its newest checkpoint (one every 50M tokens). |
| Out of memory | Should not happen on 24 GB. On a smaller card: `bash research/gpu_run.sh --skip-smoke` after lowering `--store-batch-size` in a direct `train_sae.py` call. |
| Training is far slower than 30k tokens/s | Check `nvidia-smi`: if the GPU is idle, the dataset stream is the bottleneck (slow host network). Move to another pod. |
| A labeling batch expires or requests fail | `uv run python autointerp.py --run $RUN --submit --retry-failed` sends only the failed concepts again. |
| You want to start over | Delete `research/runs/<name>/` (or just the step's output: `train_summary.json`, `examples/`, `eval.json`, `autointerp/`). |
