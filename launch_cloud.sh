#!/usr/bin/env bash
# Launch the deep_research_agent rLLM/veRL run. Presets live in config.py; this picks
# one and runs under tmux so an SSH drop doesn't kill the job (RUNPOD_PLAYBOOK pattern).
#
#   bash launch_cloud.sh sanity   # 0.5B version/import spike, a few steps
#   bash launch_cloud.sh cloud    # Qwen2.5-3B real run
#
# ⚠️  Run the SANITY preset FIRST on any fresh pod — the cheap check that the
#     rLLM<->verl<->vLLM<->torch stack trains one step together before you burn hours.
#     Then run the MASKING gate (python train_dr.py --mask-check sanity) — non-optional.
set -euo pipefail

PRESET="${1:-cloud}"
SESSION="dr_verl_${PRESET}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ -d "$HERE/.venv-deep-research" ]]; then
  # shellcheck disable=SC1091
  source "$HERE/.venv-deep-research/bin/activate"
fi

CMD="python train_dr.py ${PRESET} 2>&1 | tee run_${PRESET}.log"

if command -v tmux >/dev/null 2>&1; then
  echo "Launching in tmux session '${SESSION}'. Attach: tmux attach -t ${SESSION}"
  tmux new-session -d -s "${SESSION}" "${CMD}"
  tmux ls
else
  echo "tmux not found — running in the foreground (an SSH drop WILL kill this)."
  eval "${CMD}"
fi
