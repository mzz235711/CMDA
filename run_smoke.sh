#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${CMDA_PYTHON:-python3}"
if [[ "${1:-}" == "--setup" ]]; then
  "$python_bin" -m venv "$repo_dir/.venv"
  python_bin="$repo_dir/.venv/bin/python"
  "$python_bin" -m pip install -r "$repo_dir/requirements.txt"
fi
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cmda-matplotlib}"
cd "$repo_dir/mscn"
"$python_bin" -u run_multiseed.py --datasets job-light-ranges --methods adapt,cmda --seeds 0 --epochs 1 --device cpu --output_root "$repo_dir/results/clone_smoke/cardinality"
cd "$repo_dir/BaoForPostgreSQL/bao_server"
"$python_bin" -u run_cost_multiseed.py --datasets job-light-ranges --methods adapt,cmda --seeds 0 --epochs 1 --device cpu --output_root "$repo_dir/results/clone_smoke/cost"
