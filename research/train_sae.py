"""Train Prizma's own sparse autoencoder on GPT-2's layer-6 residual stream with SAELens.

TopK SAE, 16,384 concepts, k = 32, on ``blocks.6.hook_resid_pre`` (the tensor part A hands to the
browser), ~200M tokens of OpenWebText.

    uv run python train_sae.py                    # the real run: needs a GPU and huggingface.co
    uv run python train_sae.py --tiny             # ~1M tokens with real GPT-2: smoke test on the GPU box
    uv run python train_sae.py --random --tiny    # offline CPU test, no downloads, a few seconds

Writes ``runs/<name>/`` in SAELens format (``cfg.json``, ``sae_weights.safetensors``), which is what
``collect_top.py``, ``eval.py`` and the web export read, plus ``train_log.jsonl`` and
``train_summary.json``. See docs/gpu-runbook.md for the full sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from prizma_research.common import RUNS_DIR, add_model_args, fill_defaults, load_lm_from_args, write_json
from prizma_research.data import add_data_args, as_hf_dataset
from prizma_research.lm import EOT_ID, LM, HookedGPT2
from prizma_research.sae import SAE

FULL = dict(d_sae=16384, k=32, training_tokens=200_000_000, context_size=128, batch_tokens=4096,
            store_batch_size=256, buffer_tokens=524_288, lr_warm_up_steps=1000, dead_feature_window=1000,
            norm_batches=200, checkpoints=4, log_every=100)
TINY = dict(d_sae=2048, training_tokens=1_000_000, store_batch_size=32, buffer_tokens=65_536, lr_warm_up_steps=50,
            dead_feature_window=100, norm_batches=20, checkpoints=0, log_every=20)
RANDOM_TINY = dict(d_sae=512, k=8, training_tokens=300_000, context_size=32, batch_tokens=1024,
                   store_batch_size=16, buffer_tokens=8_192, lr_warm_up_steps=20, dead_feature_window=50,
                   norm_batches=10)


def default_run_name(lm: LM, args: argparse.Namespace) -> str:
    name = f"{lm.name}-l{lm.layer}-topk{args.d_sae}-k{args.k}"
    return f"{name}-tiny" if args.tiny and not args.random else name


class StepLogger:
    """Wraps ``sae.training_forward_pass`` to keep a JSONL log of the run without Weights & Biases."""

    def __init__(self, sae, path: Path, every: int, batch_tokens: int, total_tokens: int):
        self.inner = sae.training_forward_pass
        self.path = path
        self.every = every
        self.batch_tokens = batch_tokens
        self.total_tokens = total_tokens
        self.step = 0
        self.started: float | None = None
        self.last: dict = {}
        path.write_text("")

    def __call__(self, step_input):
        out = self.inner(step_input=step_input)
        if self.started is None:
            self.started = time.time()  # the first step includes warm-up costs; time from here
            self.first_step = self.step
        self.step += 1
        if self.step % self.every == 0:
            with torch.no_grad():
                x, x_hat = out.sae_in.float(), out.sae_out.float()
                residual = (x - x_hat).pow(2).sum(-1).mean()
                total = (x - x.mean(0)).pow(2).sum(-1).mean()
                acts = out.feature_acts.to_dense() if out.feature_acts.is_sparse else out.feature_acts
                tokens = self.step * self.batch_tokens
                elapsed = time.time() - self.started
                self.last = {
                    "step": self.step,
                    "tokens": tokens,
                    "loss": float(out.loss),
                    **{name: float(value) for name, value in out.losses.items()},
                    "explained_variance": float(1 - residual / total),
                    "l0": float((acts != 0).sum(-1).float().mean()),
                    "dead_features": int(step_input.dead_neuron_mask.sum()) if step_input.dead_neuron_mask is not None else 0,
                    "tokens_per_second": (self.step - self.first_step - 1) * self.batch_tokens / max(elapsed, 1e-9),
                    "elapsed_seconds": elapsed,
                }
            with self.path.open("a") as f:
                f.write(json.dumps(self.last) + "\n")
            print(f"[train] step {self.step:>6}  {tokens / 1e6:8.2f}M tok  mse {self.last.get('mse_loss', 0):.4f}  "
                  f"EV {self.last['explained_variance']:.3f}  L0 {self.last['l0']:.1f}  "
                  f"dead {self.last['dead_features']}  {self.last['tokens_per_second'] / 1e3:.1f}k tok/s", flush=True)
        return out


@torch.no_grad()
def check_saved_sae(run_dir: Path, lm: LM, hooked: HookedGPT2, dataset, sae_lens_sae, device: str) -> dict:
    """Reload the saved SAE with our minimal implementation and compare it with SAELens on fresh activations."""
    ours = SAE.load(run_dir).to(device)
    rows = []
    for row in dataset:
        rows.append(row["input_ids"])
        if len(rows) == 16:
            break
    tokens = torch.tensor(rows, dtype=torch.long, device=device)
    resid = hooked(tokens)
    keep = tokens != EOT_ID
    x = resid[keep]

    acts = ours.encode(x)
    x_hat = ours.decode(acts)
    reference = sae_lens_sae.to(device).eval()
    ref_acts = reference.encode(x)
    ref_hat = reference.decode(ref_acts)
    residual = (x - x_hat).pow(2).sum(-1).mean()
    total = (x - x.mean(0)).pow(2).sum(-1).mean()
    return {
        "n_tokens": int(x.shape[0]),
        "explained_variance": float(1 - residual / total),
        "l0": float((acts != 0).sum(-1).float().mean()),
        "max_abs_diff_vs_saelens_acts": float((acts - ref_acts).abs().max()),
        "max_abs_diff_vs_saelens_recon": float((x_hat - ref_hat).abs().max()),
        "mean_resid_norm": float(x.norm(dim=-1).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    add_data_args(ap, default_files=None)
    ap.add_argument("--name", default=None, help="run name; outputs go to <out>/<name>")
    ap.add_argument("--out", type=Path, default=RUNS_DIR)
    ap.add_argument("--d-sae", type=int, default=None, help="number of concepts (default 16384)")
    ap.add_argument("--k", type=int, default=None, help="active concepts per token (default 32)")
    ap.add_argument("--training-tokens", type=int, default=None, help="default 200M")
    ap.add_argument("--context-size", type=int, default=None, help="tokens per context incl. BOS (default 128)")
    ap.add_argument("--batch-tokens", type=int, default=None, help="SAE batch size in tokens (default 4096)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-warm-up-steps", type=int, default=None)
    ap.add_argument("--lr-decay-frac", type=float, default=0.2, help="linear decay over this last part of training")
    ap.add_argument("--aux-loss-coefficient", type=float, default=1.0, help="weight of the dead-concept (AuxK) loss")
    ap.add_argument("--dead-feature-window", type=int, default=None,
                    help="steps without firing before a concept counts as dead")
    ap.add_argument("--store-batch-size", type=int, default=None, help="contexts per GPT-2 forward pass")
    ap.add_argument("--buffer-tokens", type=int, default=None,
                    help="activations held in the shuffle buffer (default 524288, about 1.6 GB)")
    ap.add_argument("--norm-batches", type=int, default=None, help="batches used to estimate the activation norm")
    ap.add_argument("--autocast", action="store_true", help="bfloat16 autocast for the SAE (faster, less exact)")
    ap.add_argument("--autocast-lm", action="store_true", help="bfloat16 autocast for GPT-2 (faster, less exact)")
    ap.add_argument("--checkpoints", type=int, default=None, help="checkpoints during training (default 4)")
    ap.add_argument("--resume-from", default=None, help="checkpoint folder to resume from")
    ap.add_argument("--wandb", default=None, metavar="PROJECT", help="also log to this Weights & Biases project")
    ap.add_argument("--log-every", type=int, default=None, help="steps between log lines")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    fill_defaults(args, FULL, TINY, RANDOM_TINY)

    from sae_lens import (
        LanguageModelSAERunnerConfig,
        LanguageModelSAETrainingRunner,
        LoggingConfig,
        TopKTrainingSAEConfig,
    )

    lm = load_lm_from_args(args)
    device = str(lm.device)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")  # TF32 matmuls: free speed on recent NVIDIA GPUs
    run_dir = args.out / (args.name or default_run_name(lm, args))
    run_dir.mkdir(parents=True, exist_ok=True)
    total_steps = args.training_tokens // args.batch_tokens
    print(f"Training {run_dir.name}: TopK SAE {lm.d_model} -> {args.d_sae}, k={args.k}, on {lm.hook_name} of {lm.name}; "
          f"{args.training_tokens / 1e6:.1f}M tokens = {total_steps} steps on {device}")

    hooked = HookedGPT2(lm.gpt, lm.layer)
    dataset = as_hf_dataset(lm, context_size=args.context_size, dataset=args.dataset, data_files=args.data_files,
                            seed=args.seed)
    cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(
            d_in=lm.d_model,
            d_sae=args.d_sae,
            k=args.k,
            aux_loss_coefficient=args.aux_loss_coefficient,
            # Scale activations so their mean norm is sqrt(d_in) during training; SAELens folds the
            # factor into the weights at the end, so the saved SAE takes the raw residual stream.
            normalize_activations="expected_average_only_in",
            apply_b_dec_to_input=True,
        ),
        model_name=lm.name,
        hook_name=lm.hook_name,
        # Recorded in cfg.json: this SAE reads the raw Hugging Face residual, not a centered one.
        model_from_pretrained_kwargs={"center_writing_weights": False},
        dataset_path="synthetic" if lm.random else args.dataset,
        is_dataset_tokenized=True,
        streaming=True,
        context_size=args.context_size,
        prepend_bos=True,  # as_hf_dataset already puts <|endoftext|> first; this records it in cfg.json
        exclude_special_tokens=[EOT_ID],  # do not train on the BOS position (a huge-norm attention sink)
        training_tokens=args.training_tokens,
        train_batch_size_tokens=args.batch_tokens,
        store_batch_size_prompts=args.store_batch_size,
        # SAELens sizes its shuffle buffer as n_batches_in_buffer * context_size activations.
        n_batches_in_buffer=max(args.buffer_tokens, args.batch_tokens) // args.context_size + 1,
        n_batches_for_norm_estimate=args.norm_batches,
        lr=args.lr,
        lr_scheduler_name="constant",
        lr_warm_up_steps=min(args.lr_warm_up_steps, max(total_steps // 10, 1)),
        lr_decay_steps=int(total_steps * args.lr_decay_frac),
        dead_feature_window=args.dead_feature_window,
        feature_sampling_window=2 * args.dead_feature_window,
        autocast=args.autocast,
        autocast_lm=args.autocast_lm,
        device=device,
        dtype="float32",
        seed=args.seed,
        n_checkpoints=args.checkpoints,
        checkpoint_path=str(run_dir / "checkpoints"),
        resume_from_checkpoint=args.resume_from,
        output_path=str(run_dir),
        logger=LoggingConfig(log_to_wandb=args.wandb is not None, wandb_project=args.wandb or "prizma",
                             run_name=run_dir.name),
        verbose=False,
    )
    runner = LanguageModelSAETrainingRunner(cfg, override_dataset=dataset, override_model=hooked)
    # SAELens' built-in evaluator needs a TransformerLens model. Loss-recovered and the other
    # model-level metrics come from eval.py instead.
    runner.evaluator = None
    step_logger = StepLogger(runner.sae, run_dir / "train_log.jsonl", args.log_every, args.batch_tokens,
                             args.training_tokens)
    runner.sae.training_forward_pass = step_logger

    started = time.time()
    sae = runner.run()
    wall = time.time() - started

    check = check_saved_sae(run_dir, lm, hooked, dataset, sae, device)
    speed = step_logger.last.get("tokens_per_second")
    summary = {
        "name": run_dir.name,
        "model": lm.name,
        "hook": lm.hook_name,
        "architecture": "topk",
        "d_in": lm.d_model,
        "d_sae": args.d_sae,
        "k": args.k,
        "training_tokens": args.training_tokens,
        "context_size": args.context_size,
        "dataset": cfg.dataset_path,
        "prepend_bos": True,
        "center_input": False,
        "lr": args.lr,
        "batch_tokens": args.batch_tokens,
        "seed": args.seed,
        "device": device,
        "wall_seconds": wall,
        "tokens_per_second": speed,
        "last_logged_step": step_logger.last,
        "check_after_reload": check,
        "versions": _versions(),
    }
    write_json(run_dir / "train_summary.json", summary)

    print(f"\nSaved {run_dir} in {wall / 60:.1f} min")
    print(f"Reloaded with the minimal SAE: explained variance {check['explained_variance']:.3f}, "
          f"L0 {check['l0']:.1f}, max difference from SAELens {check['max_abs_diff_vs_saelens_recon']:.2e}")
    if speed:
        print(f"Speed {speed / 1e3:.1f}k tokens/s"
              + (f": 200M tokens take {200e6 / speed / 3600:.2f} h at this rate." if args.d_sae >= FULL["d_sae"]
                 else " (a smaller SAE than the real one, so not a forecast for the full run)."))
    if check["max_abs_diff_vs_saelens_recon"] > 1e-2 * max(check["mean_resid_norm"], 1.0):
        print("ERROR: the minimal SAE does not reproduce SAELens on this checkpoint", file=sys.stderr)
        return 1
    return 0


def _versions() -> dict:
    import sae_lens
    import transformers

    return {"sae_lens": sae_lens.__version__, "torch": torch.__version__, "transformers": transformers.__version__}


if __name__ == "__main__":
    sys.exit(main())
