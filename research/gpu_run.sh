#!/usr/bin/env bash
# Phase 1 on a rented GPU, in one command (see docs/gpu-runbook.md):
#
#   bash research/gpu_run.sh
#
# Installs uv and the Python environment, checks the GPU, then starts run_phase1.py in the
# background and follows its log. Closing the terminal does not stop the run. Running the script
# again while a run is going only follows the log; after a crash it continues where it stopped.
# Extra arguments go to run_phase1.py, e.g. `bash research/gpu_run.sh --skip-smoke`.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f phase1.pid ]] && kill -0 "$(cat phase1.pid)" 2>/dev/null; then
  echo "A run is already going (pid $(cat phase1.pid)). Following phase1.log; Ctrl-C stops following, not the run."
  exec tail --pid="$(cat phase1.pid)" -n 50 -f phase1.log
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || { echo "No NVIDIA GPU found. Rent a GPU pod first (docs/gpu-runbook.md)." >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), 'PyTorch does not see the GPU'; print('PyTorch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

nohup uv run python run_phase1.py "$@" >> phase1.log 2>&1 &
echo $! > phase1.pid
echo "Started run_phase1.py (pid $!). Following phase1.log; Ctrl-C stops following, not the run."
exec tail --pid=$! -n +1 -f phase1.log
