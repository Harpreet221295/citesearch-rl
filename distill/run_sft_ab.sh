#!/usr/bin/env bash
# 2026-09-07 — the whole pre-RL chain, unattended, in order. Stops at the first failure.
#   tests -> GRPO sanity spike -> validate teacher data -> build A + P datasets
#   -> train both (2 epochs) -> eval both on sft_dev (n=128, greedy)
# Logs: sanity_spike.log, distill/runs/<name>/train.log, distill/eval_sft_<name>_dev.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$HERE"
source .venv-deep-research/bin/activate
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

# SKIP_TO=3 skips tests + sanity (both already passed on this pod, 2026-09-07 19:50-20:0x).
SKIP_TO="${SKIP_TO:-1}"
if [[ "$SKIP_TO" -le 1 ]]; then
stamp "1/6 tests"
python -m pytest -q tests/ distill/tests/ 2>&1 | tail -3
fi
if [[ "$SKIP_TO" -le 2 ]]; then
stamp "2/6 GRPO sanity spike (never run on this pod)"
python train_dr.py sanity > sanity_spike.log 2>&1 || { stamp "SANITY FAILED — see sanity_spike.log"; tail -30 sanity_spike.log; exit 1; }
grep -E "Final validation metrics|step:2" sanity_spike.log | tail -2 || { stamp "SANITY: no step:2/final-val line — inspect sanity_spike.log"; exit 1; }
# (bug #9, 2026-09-07: this used `| xargs stamp`, but xargs cannot see a shell FUNCTION ->
#  exit 127 -> set -e killed the chain right after a PASSING sanity spike. 35 min lost.)
n_tb="$(grep -ciE "traceback" sanity_spike.log || true)"; stamp "sanity tracebacks: $n_tb"
fi

stamp "3/6 validate teacher trajectories (\$0)"
python distill/validate.py 2>&1 | tail -8

stamp "4/6 build datasets"
python distill/build_sft.py --tiers A --out distill/sft_dataset_A.pt 2>&1 | tail -14
python distill/build_sft.py --tiers P --out distill/sft_dataset_P.pt 2>&1 | tail -16

train() {  # name dataset
  local name="$1" ds="$2"; mkdir -p "distill/runs/$name"
  stamp "train $name"
  if ! python distill/sft_train.py --dataset "$ds" --batch-size 2 --grad-accum 8 --epochs 2 \
        --run-name "$name" > "distill/runs/$name/train.log" 2>&1; then
    if grep -q "OutOfMemoryError" "distill/runs/$name/train.log"; then
      stamp "OOM at batch 2 — retrying $name at batch 1 x accum 16"
      python distill/sft_train.py --dataset "$ds" --batch-size 1 --grad-accum 16 --epochs 2 \
        --run-name "$name" > "distill/runs/$name/train.log" 2>&1
    else
      tail -20 "distill/runs/$name/train.log"; exit 1
    fi
  fi
  grep -E "val_loss|adapter ->" "distill/runs/$name/train.log"
}
stamp "5/6 train"
train sft_A1210 distill/sft_dataset_A.pt
train sft_P2692 distill/sft_dataset_P.pt

stamp "6/6 eval on sft_dev"
for name in sft_A1210 sft_P2692; do
  python distill/eval_sft.py --adapter "distill/runs/$name/final" --split sft_dev --n 128 \
      --out "distill/eval_sft_${name}_dev.json" > "distill/runs/$name/eval_dev.log" 2>&1 \
      || { tail -20 "distill/runs/$name/eval_dev.log"; exit 1; }
  stamp "== $name =="; sed -n '/correct (exact match)/,/GATE/p' "distill/runs/$name/eval_dev.log" | head -30
done
stamp "SFT A/B CHAIN DONE"
