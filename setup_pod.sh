#!/usr/bin/env bash
# One-shot pod bootstrap for the deep_research_agent rLLM/veRL run.
# Idempotent-ish: safe to re-run. Follows RUNPOD_PLAYBOOK.md.
#
# The HARD part is NOT the code — it's resolving a MUTUALLY-COMPATIBLE version matrix
# (rllm <-> verl <-> vllm <-> torch <-> transformers <-> peft). That is the single
# biggest source of wasted pod hours. This installs, then FREEZES the resolved versions
# to pip-freeze.txt so the working matrix is captured forever.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv-deep-research"

echo "== 1. python venv =="
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

echo "== 2. foundation deps (mechanics + data) =="
# 2026-09-07 (fresh pod): python-dotenv must be here too — tests/test_splits.py imports
# it, and step 3's pytest gate runs BEFORE the later `pip install ... python-dotenv`,
# so the script died at step 3 on a fresh pod (6 ModuleNotFoundError errors under
# `set -euo pipefail`). Found for real 2026-09-07.
pip install pytest "datasets>=2.19" python-dotenv

echo "== 3. sanity the FRAMEWORK-FREE parts BEFORE touching rllm/verl =="
# These need no GPU/rllm and catch most bugs cheaply.
pushd "$HERE" >/dev/null
python -m pytest -q tests/            # 36 mechanics/reward/eval tests
python -c "import data; data.selfcheck()"        # BM25 surfaces gold evidence (offline)
python train_dr.py --dry-run sanity | tail -5     # config + overrides resolve
popd >/dev/null

echo "== 4. install verl + rLLM + inference engine =="
# RESOLVED 2026-08-24 (found on the /workspace migration re-run): `uv` isn't a
# system-wide tool on this pod image — it was pip-installed into the venv ad hoc
# during the first setup session and never captured here, so a fresh venv (e.g.
# after moving to a persistent volume) hits "uv: command not found" and dies on
# `set -euo pipefail`. Install it into THIS venv explicitly so the script is
# actually idempotent/reproducible from scratch, not just "worked last time
# because uv happened to already be on PATH."
pip install uv
# Resolved 2026-08-23 on the pod (see requirements.txt for full rationale). rLLM has no
# tag newer than v0.3.0-pre; main's [verl] extra pins verl==0.8.0 + vllm==0.22.1, but
# THREE separate installability bugs had to be worked around (each is a real dead end,
# not a maybe — hit all three in order):
#   1) plain `pip` can't build flash-attn (its setup.py imports torch at build time;
#      rLLM's pyproject has a [tool.uv.extra-build-dependencies] hook for this that only
#      fires when uv treats rLLM as the build ROOT — not when installed as a git dep).
#      Fix: use `uv`, and pre-install torch into the venv before the main install so
#      `--no-build-isolation` lets flash-attn's build see it.
#   2) rLLM main pins verl==0.8.0, whose BASE (unconditional) dep is numpy<2.0.0, while
#      vllm==0.22.1 transitively needs numpy>=2 (via opencv-python-headless>=4.13.0.90).
#      verl==0.9.0 (latest PyPI) fixed this (numpy>=2.0.0 + vllm>=0.18.0) but rLLM's pin
#      hasn't caught up. Fix: force verl==0.9.0 via `uv --override`.
#   3) `--no-build-isolation` (needed for #1) also starves OTHER source builds of their
#      normal auto-fetched build deps — rllm-model-gateway (an rllm subpackage) needs
#      `hatchling` at build time. Fix: pre-install hatchling too.
#   4) (found 2026-08-25, fresh pod) flash-attn isn't version-pinned here, so it resolves
#      to whatever's latest on PyPI at install time (v2.8.3 as of this fix) — that version's
#      setup.py imports psutil at build time but doesn't declare it as a build dep, and
#      --no-build-isolation means it's not auto-fetched either. Same root cause as #3.
#   5) (found 2026-08-25, fresh pod, 128-vCPU box) flash-attn's build auto-parallelizes
#      ninja to `nproc` (128 here), each nvcc job compiling one kernel (73 total). The
#      heaviest backward kernels (hdim128/256) peak at several GB of RAM each; at full
#      parallelism this blew past the container's actual CGROUP memory limit (~233GB —
#      NOT the ~2TB `free -h` reports, which is the host's, not this container's quota)
#      and the OOM killer took out dozens of compile jobs mid-build. Fix: cap parallelism
#      via `MAX_JOBS`, the env var torch's `cpp_extension` build already respects for
#      exactly this.
uv pip install torch==2.11.0 torchvision==0.26.0 --torch-backend=cu128
#   6) (found 2026-08-26, fresh pod) `ninja` was ALSO missing from the venv — same
#      --no-build-isolation starvation as #3/#4. Without it, torch's cpp_extension
#      SILENTLY falls back to serial distutils compilation instead of erroring: the
#      build looks like it's progressing normally (nvcc runs, .o files appear) but
#      compiles flash-attn's 73 CUDA kernels ONE AT A TIME at ~3 min each — ~3.5h
#      instead of ~10 min. Verified directly: only a single nvcc/cicc pair was ever
#      running despite MAX_JOBS=16, and `import ninja` failed in the venv. Note this
#      also means bug #5's MAX_JOBS cap only does anything when ninja is present —
#      the two fixes are a pair, don't keep one without the other.
uv pip install hatchling psutil ninja
# Pin every framework version actually verified to work, not just verl — bug #4 above is
# exactly what happens when a sub-dependency (flash-attn) is left to float: it silently
# resolved to a newer release between sessions and broke on a new, undeclared build dep.
# vllm is only transitively bounded (verl==0.9.0 requires >=0.18.0) — pin it too so a
# future rLLM/verl bump can't silently pull a different, unverified vllm either.
cat > /tmp/uv_overrides.txt <<'EOF'
verl==0.9.0
flash-attn==2.8.3
vllm==0.22.1
EOF
MAX_JOBS=16 uv pip install "rllm[verl] @ git+https://github.com/rllm-org/rllm.git@9beb6e0f676a46d38858991fd79ac5f8e0b16d4c" \
    --override /tmp/uv_overrides.txt --no-build-isolation
pip install wandb python-dotenv huggingface_hub
# liger-kernel: optional at the code level (verl lazy-imports it only when
# cfg.verl_use_liger=True — see ONE_STEP_TUNING_VERL_RLLM.md §9), but
# cloud_preset() sets that True by default as of 2026-08-24, so install it
# unconditionally here rather than let a real run crash on a missing import.
pip install liger-kernel

# 4th bug (found wiring the eval/checkpoint harness, 2026-08-23): vllm==0.22.1's compiled
# `_C` extension is linked against CUDA 13's runtime (`libcudart.so.13`) even though torch
# itself is a cu128 build — the .so IS present (site-packages/nvidia/cu13/lib/, pulled in
# transitively) but not on the linker path, so `import vllm` fails with
# "ImportError: libcudart.so.13: cannot open shared object file". Bake the fix into the
# venv's own activate script so every future `source .venv-deep-research/bin/activate`
# has it, instead of needing to remember an extra export every session.
NVIDIA_CU13_LIB="$VENV/lib/python3.11/site-packages/nvidia/cu13/lib"
if ! grep -q "nvidia/cu13/lib" "$VENV/bin/activate"; then
    echo "export LD_LIBRARY_PATH=\"$NVIDIA_CU13_LIB:\${LD_LIBRARY_PATH:-}\"" >> "$VENV/bin/activate"
fi
export LD_LIBRARY_PATH="$NVIDIA_CU13_LIB:${LD_LIBRARY_PATH:-}"

echo "== 5. secrets (.env) =="
ENV_FILE="$HERE/.env"
touch "$ENV_FILE"
grep -q '^HF_TOKEN=' "$ENV_FILE"      || { read -rp "HF_TOKEN: " t; echo "HF_TOKEN=$t" >> "$ENV_FILE"; }
grep -q '^WANDB_API_KEY=' "$ENV_FILE" || { read -rp "WANDB_API_KEY (wandb.ai/authorize): " w; echo "WANDB_API_KEY=$w" >> "$ENV_FILE"; }

echo "== 6. FREEZE the resolved version matrix (the valuable artifact) =="
pip freeze > "$HERE/pip-freeze.txt"
echo "   wrote pip-freeze.txt — commit it so the working matrix is captured."

echo
echo "Done. NEXT (see HANDOFF.md):"
echo "   bash $HERE/launch_cloud.sh sanity      # version/import spike + one training step"
echo "   python train_dr.py --mask-check sanity # MANDATORY masking gate"
echo "   bash $HERE/launch_cloud.sh cloud       # the real Qwen2.5-3B run"
