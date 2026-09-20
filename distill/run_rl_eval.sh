#!/usr/bin/env bash
# 2026-09-08 — post-run eval pipeline for rl_from_sft_correct_only. Waits for the training
# process (incl. its end-of-run push_checkpoints, same PID) to exit and the GPU to free,
# then: (1) merge every checkpoint (CPU) -> (2) sft_dev n=128, all checkpoints, pick the
# best by correct-and-properly-cited -> (3) heldout_eval n=300 ONCE, SFT vs best (+final)
# -> (4) musique_dev n=300, SFT vs best. One vLLM engine per eval, greedy, LoRA hot-swap.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
source .venv-deep-research/bin/activate
RUN=runs/deep_research_agent_rl_from_sft_correct_only
PID="${1:?train PID}"
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

stamp "waiting for train PID $PID to exit"
until ! kill -0 "$PID" 2>/dev/null; do sleep 20; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 500 ]; do sleep 10; done
stamp "GPU free; training log tail:"; grep -E "Final validation|\[push\] done|\[push\]   step" run_rl_from_sft.log | tail -8 | cut -c1-200

stamp "1/4 merge checkpoints (CPU)"
# ISSUE #18 (2026-09-08): the first version captured this block's stdout as the adapter
# list, and hub.merge_checkpoint PRINTS its "exited non-zero but adapter written" note to
# stdout -> eval_rl.py received that sentence as arguments and died; the monitor's filter
# matched "Error" but not argparse's lowercase "error:". 30 min lost. Diagnostics now go
# to stderr; only the adapter list reaches stdout.
ADAPTERS=$(CUDA_VISIBLE_DEVICES="" python - <<'PY'
import sys, contextlib
from pathlib import Path
import hub
run = Path("runs/deep_research_agent_rl_from_sft_correct_only")
out = []
with contextlib.redirect_stdout(sys.stderr):
    for s in hub.list_checkpoint_steps(run):
        d = run / "_merged" / f"step{s}" / "lora_adapter"
        if not (d / "adapter_model.safetensors").exists():
            hub.merge_checkpoint(run / f"global_step_{s}", run / "_merged" / f"step{s}")
        out.append(f"--adapter step{s}={d}")
print(" ".join(out))
PY
)
case "$ADAPTERS" in --adapter*) ;; *) stamp "adapter list looks wrong: $ADAPTERS"; exit 1;; esac
echo "   $ADAPTERS"

stamp "2/4 sft_dev n=128 — SFT vs every checkpoint"
python distill/eval_rl.py --split sft_dev --n 128 $ADAPTERS --out distill/eval_rl_sft_dev.json > distill/eval_rl_sft_dev.log 2>&1 || { tail -20 distill/eval_rl_sft_dev.log; exit 1; }
sed -n '/correct (exact match)/,/produced an answer/p' distill/eval_rl_sft_dev.log
BEST=$(python - <<'PY'
import json
d = json.load(open("distill/eval_rl_sft_dev.json"))["summary"]
cands = {k: v for k, v in d.items() if k != "sft"}
best = max(cands, key=lambda k: (cands[k]["correct_and_cited_rate"], cands[k]["correct_rate"]))
print(best)
PY
)
LAST=$(echo "$ADAPTERS" | grep -oE "step[0-9]+=" | tail -1 | tr -d '=')
stamp "best on sft_dev by correct-and-properly-cited: $BEST (final checkpoint: $LAST)"
BEST_PATH=$(echo "$ADAPTERS" | tr ' ' '\n' | grep "^$BEST=" | cut -d= -f2)
LAST_PATH=$(echo "$ADAPTERS" | tr ' ' '\n' | grep "^$LAST=" | cut -d= -f2)
EXTRA=""; [ "$BEST" != "$LAST" ] && EXTRA="--adapter $LAST=$LAST_PATH"

stamp "3/4 heldout_eval n=300 — ONCE: SFT vs $BEST ${EXTRA:+(+ $LAST)}"
python distill/eval_rl.py --split heldout_eval --n 300 --adapter "$BEST=$BEST_PATH" $EXTRA --out distill/eval_rl_heldout.json > distill/eval_rl_heldout.log 2>&1 || { tail -20 distill/eval_rl_heldout.log; exit 1; }
sed -n '/correct (exact match)/,/GATE/p' distill/eval_rl_heldout.log | head -60

stamp "4/4 musique_dev n=300 — SFT vs $BEST"
python distill/eval_rl.py --split musique_dev --n 300 --adapter "$BEST=$BEST_PATH" --out distill/eval_rl_musique_dev.json > distill/eval_rl_musique_dev.log 2>&1 || { tail -20 distill/eval_rl_musique_dev.log; exit 1; }
sed -n '/correct (exact match)/,/BY HOP/p' distill/eval_rl_musique_dev.log | head -40
stamp "RL EVAL PIPELINE DONE"
