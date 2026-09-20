# rLLM + veRL install — the resolved version matrix, and every dead end it took to get there

**Date:** 2026-08-23. **Pod:** RunPod 1× A100 80GB SXM4, driver 580.159.03 (CUDA 13.0 capable),
system CUDA toolkit 12.4 (`/usr/local/cuda`), 255 CPU cores, 2TB RAM, 60GB disk, Python 3.11.10.

**Status: install fully DONE and verified.** `import torch, rllm, vllm, verl, flash_attn` all
succeed, `torch.cuda.is_available() == True`, and `rllm.environments.base.base_env.BaseEnv` /
`rllm.trainer.agent_trainer.AgentTrainer` both import cleanly. `pip-freeze.txt` is committed —
304 packages, ground truth over anything narrated below if the two ever disagree. Disk after
install: 29GB used / 60GB (plenty of headroom for model weights + checkpoints).

**Update 2026-08-23 (later same day), wiring the eval/checkpoint/W&B harness:** found two
MORE real bugs beyond the four install-time ones (bugs 4 and 5 below — a `libcudart.so.13`
linker-path gap, and a sequential-vLLM-engine GPU-memory-not-actually-freed race). Both fixed
and verified end-to-end on the pod: `evaluate.py`'s batched-vLLM path + `judge.py`'s vLLM
backend + W&B eval logging all ran for real against a live model on this A100. See
`HANDOFF.md` §1 for the current per-file status. Still genuinely open: the `AgentTrainer`
`workflow_class` port (HANDOFF.md step 2) — confirmed broken, loudly flagged in `train_dr.py`,
not yet fixed; `hub.py`'s checkpoint push is therefore mechanically ready but untested against
a real trained checkpoint.

**Why this file exists:** HANDOFF.md step 1 ("resolve the version matrix") turned out to be
almost the ENTIRE session — six install attempts, three distinct real bugs, one red herring
that ate 20 minutes of investigation. `assignments/finqa_agent/verl/`'s equivalent migration
was scaffolded but **never actually completed** (its `pip-freeze.txt` still has the rllm/verl/
vllm lines as unfilled `<TODO>`s) — so this is the first time anyone has pushed this stack all
the way to a working install in this repo. Read this before you touch rLLM/veRL again, on
*any* lab, so you don't re-spend the hour this cost.

**If you're a future session on a fresh pod:** you don't need this file's blow-by-blow — just
run `bash setup_pod.sh`, which now has every fix baked in. Read this file when something
STILL breaks (versions drift fast) or when you're doing the same resolution for a different
lab (finqa's verl migration, or a new one) and want to skip straight to what's likely to bite.

---

## TL;DR — the resolved matrix

| package | version | why this exact one |
|---|---|---|
| `rllm` | `main` @ commit `9beb6e0f676a46d38858991fd79ac5f8e0b16d4c` | no tag newer than `v0.3.0-pre` (2026-04-30) exists; main is 130 commits ahead. **Pin the commit SHA**, not `main` floating — it moves. |
| `verl` | **`0.9.0`** (overriding rLLM's own pin of `0.8.0`) | rLLM main's pyproject pins `verl==0.8.0`, which is STALE — see Issue #2 below. `0.9.0` is the latest PyPI release and is what actually installs. |
| `torch` | `2.11.0+cu128` | forced by `vllm==0.22.1`'s own exact pin (`torch==2.11.0`) — not by rLLM's loose `torch>=2.10.0` floor. |
| `torchvision` | `0.26.0+cu128` | vllm's pin. |
| `vllm` | `0.22.1` | rLLM main's pin (ships as a prebuilt wheel — no build issues). |
| `flash-attn` | `2.8.3` | rLLM main's pin. **No prebuilt wheel for torch 2.11 exists yet** (GitHub release wheels only go up to `cu12torch2.8`) → builds from source. Fine on this pod (255 cores, ~4 min compile), but see Issue #1/#3. |
| `transformers` | `>=5.5.3` (rLLM's floor; resolves to latest) | |
| `hatchling` | latest, pre-installed manually | build dep of `rllm-model-gateway`, an rLLM subpackage — see Issue #3. |
| CUDA toolkit used to compile flash-attn | system `/usr/local/cuda` (**12.4**) | NOT the version torch was built against (12.8) — and that mismatch is FINE (see the CUDA red herring below). |

**The exact working install sequence** (also in `setup_pod.sh` step 4 — that's the source of
truth if this doc and the script ever disagree):
```bash
uv pip install torch==2.11.0 torchvision==0.26.0 --torch-backend=cu128   # BEFORE anything else
uv pip install hatchling
echo "verl==0.9.0" > /tmp/uv_overrides.txt
uv pip install "rllm[verl] @ git+https://github.com/rllm-org/rllm.git@9beb6e0f676a46d38858991fd79ac5f8e0b16d4c" \
    --override /tmp/uv_overrides.txt --no-build-isolation
pip install wandb python-dotenv huggingface_hub
pip freeze > pip-freeze.txt   # commit this — it's the ground truth, more than this doc is
```
Order matters: torch and hatchling MUST land in the venv **before** the big install, and
`--no-build-isolation` is required — see below for why each piece is there.

---

## The three real bugs (in the order they were hit)

### Bug 1 — plain `pip` can't build flash-attn: "No module named 'torch'"
**Symptom:** `pip install "rllm[verl] @ git+..."` fails while building `flash-attn`:
```
Getting requirements to build wheel: finished with status 'error'
...
File "<string>", line 22, in <module>
ModuleNotFoundError: No module named 'torch'
```
**Root cause:** flash-attn's `setup.py` imports `torch` at build time (to detect the CUDA
arch to compile for). Normal PEP-517 builds run in an *isolated* temp environment that only
has what the package's own `build-system.requires` declares — flash-attn doesn't declare
`torch` there (a known upstream wart), so the isolated env never has it.

rLLM's own `pyproject.toml` actually has the fix already written for this — a `[tool.uv.
extra-build-dependencies]` block:
```toml
[tool.uv.extra-build-dependencies]
flash-attn = [
    { requirement = "torch", match-runtime = true },
]
```
**But this only fires when `uv` treats rLLM's own repo as the build ROOT** (a local editable
`uv pip install -e .`, matching the docs' recommended flow). It does **not** fire when rLLM is
pulled in as a transitive `git+https://...` dependency of nothing-in-particular, which is what
`requirements.txt` needs to do (this lab isn't a clone-and-edit of rLLM itself).

**Fix:** don't rely on that hook at all — (a) use `uv` instead of plain `pip` in the first
place (plain pip doesn't even have an isolation-override flag that helps here without also
fully losing isolation), (b) pre-install `torch` into the venv *before* the main install, (c)
pass `--no-build-isolation` so flash-attn's build step sees the venv's own site-packages
(which now has torch) instead of an isolated temp env.

**A gotcha inside this gotcha:** the first attempt used `pip install ... | tee log.txt` in a
background shell. `tee`'s own exit code (0, it succeeded at writing the file) masked the real
failing exit code of `pip` — the task notification reported "completed (exit code 0)" for a
run that had actually failed. **Always `set -o pipefail` before piping an install command
through `tee`**, or check the log content, not just the reported exit code.

### Bug 2 — numpy conflict: `verl==0.8.0` vs `vllm==0.22.1`
**Symptom:** once bug 1's fix (uv, `--torch-backend=cu128`) was in place, resolution itself
failed before any building happened:
```
× No solution found when resolving dependencies:
  ╰─▶ Because opencv-python-headless>=4.13.0.90 depends on numpy>=2 and
      vllm==0.22.1 depends on opencv-python-headless>=4.13.0, we can conclude
      that vllm==0.22.1 depends on numpy>=2.
      And because verl==0.8.0 depends on numpy<2.0.0, ...
```
**Root cause:** rLLM main's pyproject pins `verl==0.8.0` **exactly**. Checked verl 0.8.0's own
PyPI metadata: its BASE (unconditional, not extra-gated) dependency list has `numpy<2.0.0`.
`vllm==0.22.1` needs `opencv-python-headless>=4.13.0.90` (a real dep, for video/image
preprocessing in multimodal rollout paths), and *that* package's own modern releases all
require `numpy>=2`. Direct, unresolvable conflict as literally pinned.

Checked whether an OLDER rLLM tag avoids this: `v0.3.0-pre` pins `verl==0.7.1` — checked ITS
PyPI metadata too — **same** `numpy<2.0.0` base constraint. Not a fluke of 0.8.0 specifically.

Checked verl's PyPI version history for when this got fixed: **`verl==0.9.0`** (released after
rLLM's main branch last bumped its pin — a real case of one fast-moving repo's pin lagging
behind another) flips to `numpy>=2.0.0` in its base deps, AND widens its own `[vllm]` extra
constraint from `vllm<=0.12.0,>=0.8.5` to `vllm>=0.18.0` — meaning 0.9.0 is also the version
verl's own maintainers consider compatible with a vllm as new as 0.22.1.

**Fix:** force `verl==0.9.0` regardless of what rLLM's pyproject says, via `uv pip install
--override overrides.txt` (a file containing `verl==0.9.0`). uv's `--override` forces a
version irrespective of what any dependency in the graph requests, without editing rLLM's own
pyproject (which we don't control / isn't ours to patch).

**Lesson:** when a framework's own stated compatible-version range for a sub-dependency
(`verl==0.8.0` requiring `numpy<2.0.0`) conflicts with what a SIBLING pinned package needs,
don't assume the top-level pin (`verl==0.8.0`) is gospel — check whether a newer release of
the CONFLICTING package fixed it, especially when the top-level repo (rLLM) is itself a fast-
moving, frequently-unreleased-tag project (no tag newer than a `-pre` from 4 months ago).

### Bug 3 — `--no-build-isolation` breaks a DIFFERENT package's build: `rllm-model-gateway` needs `hatchling`
**Symptom:** after fixing bugs 1 and 2, hit a NEW error:
```
Failed to build `rllm-model-gateway @ git+...#subdirectory=rllm-model-gateway`
Call to `hatchling.build.build_wheel` failed
ModuleNotFoundError: No module named 'hatchling'
```
**Root cause:** `--no-build-isolation` (needed to fix Bug 1) is a BLUNT, GLOBAL flag — it
disables isolated builds for *every* package in the resolution that needs building from
source, not just flash-attn. `rllm-model-gateway` (a subpackage rLLM's own repo ships, pulled
in automatically as a path dependency) uses `hatchling` as its build backend. Normally that's
auto-fetched into its own isolated build env; with isolation off, it's just... not there.

**Fix:** pre-install `hatchling` into the venv too, same pattern as torch. Simple once you see
it, but non-obvious the first time — expect this class of bug (isolation-off breaking some
OTHER package's normal isolated build) to recur if new packages get added to this dependency
tree later. If it happens again: whatever `ModuleNotFoundError` names is what to pre-install.

---

## Two things that WEREN'T bugs (learn from the wasted time)

### The infra hiccup (not a version-matrix bug at all)
After bugs 1–3 were all fixed and the install got much further (built rllm, built
rllm-model-gateway, downloaded vllm's 249MB wheel + flashinfer-cubin's 344MB, `verl==0.9.0`
actually landed in site-packages) — the session itself got interrupted mid-install. The task
notification came back as `"status": "stopped"` with **"No completion record was found... may
have been stopped via the UI, Monitor timeout, or agent teardown."** This was an environment/
harness hiccup, unrelated to anything about rLLM or veRL. **Diagnostic:** don't assume a
"stopped" or missing-completion status means your fix was wrong — check what's ACTUALLY in
site-packages (`ls .venv/lib/python3.11/site-packages/ | grep -i <pkg>`, try the real
`import`) before concluding anything. In this case only `verl` had landed; re-running the
identical command picked up from the uv cache (near-instant for everything already
downloaded/built) rather than starting over.

### The CUDA-version-mismatch red herring
The RE-run then hit a genuine-looking failure:
```
RuntimeError: Error compiling objects for extension
hint: flash-attn (v2.8.3) was included because rllm[verl] depends on flash-attn
```
...but `uv`'s own output **truncated the actual compiler error** — the traceback showed
Python's build machinery unwinding, not the real `nvcc`/`gcc` diagnostic line. This is a real
limitation worth remembering: **uv/pip do not reliably surface the underlying compiler's error
text when a source build fails** — you have to go get it yourself.

Working hypothesis at the time (reasonable, wrong): the pod's system `nvcc` is CUDA **12.4**
(`/usr/local/cuda`), but torch was built against CUDA **12.8** — surely THAT'S the mismatch
breaking the compile? Spent real time chasing this:
- Looked for a pip-installable nvcc matching 12.8 exactly. Found `nvidia-cuda-nvcc-cu12==
  12.8.93` (exact version match) — but its file listing has **no `nvcc` binary at all**, only
  `ptxas`, headers, and `nvvm`/`libdevice` — a genuine NVIDIA pip-packaging gap for this
  particular release naming scheme. The only COMPLETE pip-installable nvcc found (with an
  actual `nvcc` binary) was the *unversioned* `nvidia-cuda-nvcc` package, which resolves to
  `13.3.73` (CUDA 13.3) — moving the mismatch further, not fixing it.
- Considered switching torch to a `cu124` build to match the system nvcc instead. Checked
  `download.pytorch.org/whl/cu124/torch/` for a `2.11.0` wheel — **doesn't exist**; that CUDA
  tag was retired somewhere around torch 2.6/2.7 in favor of cu126/cu128/cu129/cu130. Dead end.

**What actually resolved it:** stopped guessing and got the REAL error directly — downloaded +
extracted the flash-attn 2.8.3 source tarball manually (`pip download --no-deps --no-binary
:all:`), and ran `python setup.py build_ext` by hand with full verbose output captured to a
plain log file (not piped through `uv`'s summarizer). Two things fell out of this:
1. torch's OWN internal warning, printed right there in the log: *"The detected CUDA version
   (12.4) has a minor version mismatch with the version that was used to compile PyTorch
   (12.8). **Most likely this shouldn't be a problem.**"* — straight from the horse's mouth,
   the mismatch theory was wrong.
2. The manual build then proceeded CLEANLY — 73 objects, 32 parallel `nvcc`/`cc1plus`
   processes (this pod's 255 cores earning their keep), zero real compiler errors, done in a
   few minutes.

**Conclusion:** the earlier `uv`-driven failure was almost certainly **resource contention
from an overlapping/interrupted install attempt** (a stale partial ninja build directory left
behind by the session that got cut off mid-install, or two install processes racing on the
same build temp dir) — NOT a real toolchain incompatibility. Running flash-attn's build in a
clean, isolated directory with nothing else touching it fixed it. **Confirmed for real**: the
exact same `uv pip install ... --no-build-isolation` command that had failed twice was re-run
afterward with nothing else running, and it built flash-attn clean in one shot (~9 minutes,
`Built flash-attn==2.8.3`) — same command, same pins, only difference was a quiet machine.

**Lesson for next time something fails mysteriously mid-build:** (1) don't trust a wrapped
package manager's summarized error — go find the raw compiler output; (2) a CUDA
toolkit/torch minor-version mismatch (12.4 vs 12.8, same major) is usually a non-issue for
Ampere-class (sm_80) targets — torch says so itself; (3) if a build fails right after an
interrupted/overlapping install attempt touched the same package, suspect a stale build
artifact/contention before suspecting the pins.

---

## Bug 4 (found later, wiring the eval harness) — `vllm` needs `libcudart.so.13`
**Symptom:** `import vllm` (fine at first check, but only because that check never actually
loaded vllm's compiled extension) later fails for real:
```
ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
```
raised from `vllm/platforms/cuda.py` → `import vllm._C`.
**Root cause:** vllm 0.22.1's compiled `_C` extension is linked against **CUDA 13's**
runtime, even though torch itself is a cu128 (CUDA 12.8) build — a real split inside the
resolved dependency tree (torch pulls `nvidia-cuda-runtime-cu12`; something else, likely
vllm's own build target, transitively pulled unversioned `nvidia-cuda-runtime` at `13.3.29`).
The `.so` **is already present** at `site-packages/nvidia/cu13/lib/libcudart.so.13` — it's
simply not on the dynamic linker's search path (`LD_LIBRARY_PATH`); torch's own CUDA libs get
found via a different mechanism (RPATH patching at wheel-build time) that this sibling
package doesn't share.
**Fix:** `export LD_LIBRARY_PATH="<venv>/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"`.
Baked into `setup_pod.sh`, which now appends this line directly into the venv's own
`bin/activate` script (idempotent — checks for it first) so it's automatic on every future
`source .venv-deep-research/bin/activate`, not something to remember per-session.
**Lesson, same shape as Issue #1/#3:** a fast-moving multi-package CUDA stack can split
across TWO CUDA major versions (12.x from torch, 13.x from vllm) even when everything
*installs* without conflict — the numpy-style dependency resolver conflict (Bug 2) isn't the
only way this shows up; a working `pip`/`uv` install can still produce a binary that fails at
**import** time, not install time. Don't assume "installed cleanly" means "imports cleanly" —
actually `import` every package you depend on before trusting the environment.

## Bug 5 (runtime, not install-time) — sequential vLLM engines: `del` doesn't free GPU memory
**Symptom:** loading a SECOND vLLM engine after finishing with a first one (e.g. eval's
policy engine, then a judge engine) fails:
```
ValueError: Free memory on device cuda:0 (30.65/79.25 GiB) on startup is less than
desired GPU memory utilization (0.6, 47.55 GiB).
```
even though the first engine's Python object was `del`eted, `gc.collect()`ed, and
`torch.cuda.empty_cache()`d first.
**Root cause:** vLLM V1 runs the actual engine in a **separate subprocess** ("EngineCore",
its own PID, visible in the logs as `(EngineCore pid=NNNNN)`). `LLMEngine.__del__` (checked
the installed `vllm==0.22.1` source directly) does **not** call `engine_core.shutdown()` —
it only tears down a `dp_group`. So deleting the Python-level `LLM` handle in the parent
process never signals the child subprocess to exit; the child keeps holding its GPU memory
indefinitely (or until the whole Python process exits, which happens to trigger a proper
shutdown — this is why it looked fine in a script that only ever loads ONE engine).
**Fix:** explicitly call `llm.llm_engine.engine_core.shutdown()`, THEN **poll**
`torch.cuda.mem_get_info()` until the memory is actually back (don't trust the shutdown call
to be synchronous — it wasn't, empirically: memory was still ~48GB resident when the very
next line tried to build a second engine). `evaluate.close_vllm_engine` / `judge.Judge.close`
now do this, with a 90s timeout and a loud warning (not a silent hang) if it's exceeded.
**Lesson:** the same shape as Bug 4 — "installed cleanly" and even "ran once successfully"
doesn't mean a MULTI-PHASE usage pattern (load engine A, tear it down, load engine B) is
safe. Verify the SPECIFIC usage pattern you need, not just the happy path of one engine
for the life of the process.

## Smaller wins bundled into the same session

- **`data._load_2wiki`'s HF mirror was broken**: the hardcoded `xanhho/2WikiMultihopQA` is a
  **script-based dataset** and fails outright on `datasets>=3` ("Dataset scripts are no longer
  supported") — literally the RUNPOD_PLAYBOOK gotcha #1, hit for real. Verified `voidful/
  2WikiMultihopQA` loads cleanly and confirmed its exact row schema by inspecting a real row
  (`context: list[[title, [sentences]]]`, `supporting_facts: list[[title, sent_id]]`) — the
  parsing code already in `data.py` was actually correct for this shape; only the mirror id
  needed to change. Fixed in `data.py`.
- **Doc pages for rLLM disagreed with each other and with reality — three different times.**
  `rllm-project.readthedocs.io` said "cu128, no explicit pins." `docs.rllm-project.com` said
  verl `0.7.1`/vllm `~0.17.x`-ish plus an `AgentTrainer(agent_flow=..., evaluator=...)` shape.
  The `examples/fully_async/deepresearch/train.py` example used a THIRD API,
  `AsyncAgentTrainer(rollout_fn=..., val_rollout_fn=...)`. None of these matched what raw
  source inspection (`rllm/trainer/agent_trainer.py`'s actual `__init__` signature) showed.
  **Lesson, restated because it mattered three separate times this session: for a fast-moving
  framework, treat rendered doc pages as a starting guess, not truth — verify against the raw
  `pyproject.toml` / source file at the exact pinned commit before writing code against it.**

---

## What's still open (not this file's job — see HANDOFF.md)

This file only covers **getting the packages installed**. Still ahead per HANDOFF.md step 2+:
wiring the real `AgentTrainer` signature (confirmed different from the `env.py`/`train_dr.py`
scaffold's `agent_class`/`env_class`/`env_args` guess — real signature takes `workflow_class`/
`workflow_args`; `train_dataset`/`val_dataset` want `rllm.data.Dataset` objects, not raw
`list[dict]`), the `BaseEnv` import path (confirmed: `rllm.environments.base.base_env.BaseEnv`,
and its contract — `reset`/`step`/`from_dict`/`idx`/`close` — DOES match what `env.py` already
assumes, that part needs no rework), the hydra key names in `train_dr.verl_overrides`, and the
masking-gate rollout extraction. Update this file's "still open" section as those get resolved,
or start a sibling doc — don't let this one silently go stale.
