# Fresh-pod setup & sanity check — deep_research_agent

**Date:** 2026-08-25. **Purpose:** the fast-path playbook for going from a genuinely
fresh pod to a verified-working environment — written after a real session that hit
five distinct infra bugs doing exactly this. **Read this FIRST on any new pod**, before
`HANDOFF.md`'s research-status sections or the deeper bug-archaeology docs
(`RLLM_VERL_INSTALL_NOTES.md`, `WORKFLOW_PORT_NOTES.md`) — those explain *why* each fix
exists; this doc is the checklist to actually run.

**Pod policy, per Harpreet (2026-08-25):** **terminate pods between sessions, don't
stop/start.** No Network Volume, no cross-pod local persistence attempt — this doc
assumes you're starting from nothing every time. Code/docs persist via `git push` to
GitHub; checkpoints via HF Hub (`hub.py`); metrics via W&B. `POD_LIFECYCLE.md` (the
previous session's stop/resume-specific playbook, written for a pod with a fast local
Volume Disk) is now largely historical — kept for its real bug write-ups (the
venv-shebang gotcha below is one), not as the active plan.

---

## 1. Where to put things — local disk only, always `/root`

**Clone directly to `/root` (or equivalent local Container Disk). Never `/workspace`,
full stop — no benchmarking it, no considering it, no exceptions.** This is Harpreet's
explicit, standing call (2026-08-25): every pod from now on is terminated between
sessions, not stopped, so there's no cross-pod persistence to gain from `/workspace`
in the first place — the entire reason to consider it is gone. It's also a real
liability: `/workspace` is sometimes a local "Volume Disk" (fast) and sometimes a
genuinely network-mounted "Network Volume" (`mfs://...`, ~50x slower for the
many-small-file writes a venv/flash-attn build does), indistinguishable by path
alone — guessing wrong wasted real time in the session that led to this rule (a build
process got stuck in permanent, unkillable uninterruptible I/O wait — `D` state — on
the slow mount). Don't relitigate this per-pod; just always build in `/root`.

```bash
cd /root
git clone https://<your-PAT>@github.com/Harpreet221295/Agentic-RL-Alignment-Path.git
cd Agentic-RL-Alignment-Path/assignments/deep_research_agent
```
(Supply your own PAT directly in your own terminal — don't have an agent construct or
re-type a URL containing it; that's a credential-handling anti-pattern regardless of
who's typing it, and this session's safety classifier correctly refused it.)

---

## 2. Secrets — `.env`, gitignored, always needed fresh

```bash
# assignments/deep_research_agent/.env (create this file, never commit it)
HF_TOKEN=...
WANDB_API_KEY=...
```
Gone on every fresh pod by design (gitignored). `setup_pod.sh` will prompt for these
interactively if the file doesn't already have them — pre-populate to skip the prompt
when running it non-interactively/backgrounded.

---

## 3. Run `setup_pod.sh` — what's already fixed, what to watch for

```bash
bash setup_pod.sh
```
As of 2026-08-25 this script has, baked in:
- The full resolved version matrix (`torch==2.11.0+cu128`, `verl==0.9.0`,
  `flash-attn==2.8.3`, `vllm==0.22.1` — **all hard-pinned**, not left to float. This
  matters: an unpinned `flash-attn` silently resolved to a different release between
  sessions and broke on an undeclared build dependency — see §4 bug 3 below).
- `MAX_JOBS=16` capping the flash-attn CUDA-kernel compile's parallelism (see §4 bug 4
  — full, unconstrained parallelism can OOM-kill the build on a container whose real
  cgroup memory limit is smaller than what `free -h` reports).

**Expect the flash-attn build step to be the slow part** — several minutes even at
capped parallelism (73 kernel objects to compile). This is normal, not stuck. To check
genuine progress vs. a real hang:
```bash
# object files compiled so far (should climb over time):
find /root/.cache/uv/sdists-v9/pypi/flash-attn -name "*.o" 2>/dev/null | wc -l
# is the compiler actually running (not blocked)?
ps aux | grep -E '[n]vcc|[c]c1plus'
# real memory ceiling for THIS container (compare against .../memory.current):
cat /sys/fs/cgroup/memory.max
```
**Red flag to watch for:** `Killed` lines in the install log (`grep -c Killed
setup_pod.log`) — that's the OOM killer, meaning parallelism is still too high for this
particular pod's memory quota even with `MAX_JOBS=16`. If it recurs, lower `MAX_JOBS`
further (try 8) rather than re-running the same command hoping it's transient.

**Never `pkill -f <pattern>` to clean up a stuck process** if the pattern could also
match your own current command's text — `pkill -f` matches against the full command
line system-wide, including the shell invocation running the `pkill` itself, and will
silently kill your own calling shell. Kill by PID instead
(`ps aux | grep '[p]attern'` — bracket trick to exclude the grep line itself — then
`kill -9 <pid>`).

**The same trap applies to `pgrep`, and it has now bitten twice in two different
forms** (2026-08-25 `pkill`, 2026-08-26 `pgrep`). Any `-f` pattern match is
system-wide over full command lines, so a WAIT loop like:
```bash
until ! pgrep -f "bash setup_pod.sh" > /dev/null; do sleep 10; done   # HANGS FOREVER
```
never exits — the loop's own command line contains the literal text
`bash setup_pod.sh`, so `pgrep` matches the watcher itself and the condition stays
true after the real job is long gone. It fails SILENTLY (it just waits), which is why
it is easy to miss: six of these accumulated in one session before anyone noticed, all
watching work that had already finished. Use the bracket trick, which cannot match
itself because the literal text differs from the regex:
```bash
until ! pgrep -f "[b]ash setup_pod.sh" > /dev/null; do sleep 10; done  # correct
```
Or capture the PID at launch and wait on that instead (`kill -0 "$pid"`), which has no
pattern to self-match at all. Reading this doc is not enough to avoid it — the
2026-08-26 session read this very section and then wrote the `pgrep` form anyway.

---

## 4. The bugs hit getting here (2026-08-25 and 2026-08-26, fresh pods) — quick reference

Full detail in `setup_pod.sh`'s own inline comments (search `2026-08-25`) and
`RLLM_VERL_INSTALL_NOTES.md` for the *original* (2026-08-23) bug set this builds on.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Build process stuck forever, unkillable (`D` state) | `/workspace` was a slow network-mounted volume, not local disk | Build on local disk only (§1) |
| 2 | `bad interpreter: No such file or directory` after moving a built venv | venvs bake absolute paths into `pyvenv.cfg`/shebangs — moving breaks them | Never move a built venv; rebuild fresh at the final path |
| 3 | `ModuleNotFoundError: No module named 'psutil'` mid flash-attn build | `flash-attn` wasn't version-pinned, resolved to a newer release needing an undeclared build dep | Pin `flash-attn` exactly (now done); pre-install `psutil` |
| 4 | `RuntimeError: Error compiling objects for extension`, many `Killed` lines in the log | `ninja` auto-parallelized to `nproc` (128), OOM-killing jobs against the container's real ~233GB cgroup limit (not the ~2TB `free -h` reports) | `MAX_JOBS=16` (now baked into `setup_pod.sh`) |
| 5 | Calling shell died instantly on `pkill -f "setup_pod.sh"` | The pattern matched the `pkill` command's own invocation text, killing itself | Kill by PID, not by a self-matching pattern |
| 5b | A `until ! pgrep -f "<job>"` wait loop never exited, long after the job finished — silent, just waits | Same self-matching cause as #5: the watcher's own command line contains the pattern text | Bracket the pattern (`"[b]ash setup_pod.sh"`) or wait on a captured PID (`kill -0 "$pid"`) |
| 6 | flash-attn build looked healthy but crawled — ~3.5h instead of ~12min for 73 kernels | `ninja` missing from the venv (same `--no-build-isolation` starvation as #3/#4), so torch's `cpp_extension` SILENTLY fell back to serial distutils compilation. Only ONE nvcc ran despite `MAX_JOBS=16` | `uv pip install ... ninja` (now in `setup_pod.sh`). NOTE `MAX_JOBS` is inert without ninja — the two fixes are a pair |
| 7 (2026-09-07) | `45 passed, 6 errors` at step 3, script stops silently | `tests/test_splits.py` imports `dotenv`; step 4 installs it, step 3 runs first | `python-dotenv` moved into step 2 |
| 8 (2026-09-07, self-inflicted) | step 3's `--dry-run sanity` dies with a "local path has no config.json" message | a new guard in `train_dr.py` keyed on `os.sep in model_name`; Hub ids contain a slash too | key on `os.path.isabs()`; test guards against the inputs the gate actually uses |
| 9 (2026-09-07, ops) | an unattended driver stops right after a PASSING stage; log ends `xargs: <name>: No such file or directory` | `… \| xargs -I{} <bash-function>` — xargs cannot see shell functions → exit 127 → `set -e` | never call a shell function from xargs; dry-run every line, watch the GPU not just the log |
| 10 (2026-09-07, ops) | a relaunch watcher reports the PREVIOUS run's traceback; new process "dead" but alive | `slow_cmd && nohup new_run … &` backgrounds the whole chain, so the log is truncated only after `slow_cmd` | truncate the log as its own command first; never chain a slow foreground step in front of a backgrounded launch |
Full write-ups of all of these, plus the training-signal bug F18, are in `RL_FROM_SFT_LOG.md` §5.

---

## 5. Sanity-check the result — in order, cheapest first

Don't trust "the install script exited 0" alone. Run these, in this order:

```bash
source .venv-deep-research/bin/activate   # every new shell needs this — it does not persist across separate shell sessions

python -m pytest -q tests/ distill/tests/ # expect: 78 passed (2026-09-07: 45 + 18 + the
                                          # RL-from-SFT wiring tests in tests/test_rl_from_sft.py)
python -c "import data; data.selfcheck()" # expect: BM25 surfaces gold evidence for all 4 fixture questions
python train_dr.py --dry-run sanity       # expect: resolved config + veRL overrides print, no error

python -c "import torch, vllm, verl, flash_attn, rllm; print(torch.cuda.is_available())"
                                           # expect: True — imports alone aren't enough, some packages
                                           # (vllm especially) can install cleanly but fail at import time
```

**The real proof, not optional — a genuine end-to-end GRPO step:**
```bash
python train_dr.py sanity
```
This is the one check that actually confirms rLLM→veRL→vLLM→GRPO work together, not
just that packages import. **Budget ~4-5 minutes of wall-clock** — most of it is fixed
one-time overhead (Ray init, FSDP model load, vLLM engine warmup + CUDA graph capture
across ~170 batch-size buckets), not the training step itself. Watch it via the process
CPU state / `nvidia-smi` if the console looks quiet for a while — vLLM's graph capture
phase genuinely takes 1-3 minutes and produces little console output until the phase
transitions; that's expected, not a hang.

**What a PASSING run's tail looks like** (real numbers vary, shape shouldn't):
```
[batch-safety-patch] N collected rows not divisible by requested mini_batch_size=... Falling back to one full-batch mini-batch...
step:1 - ... - reward_components/outcome:... - reward_components/groundedness:... - batch/dead_groups_pct:... 
[EpisodeLogger] Logging N episodes for step=2, mode=val, epoch=0
("Final validation metrics: {'val/deep_research_agent/pass@1': ...}")
step:2 - val/deep_research_agent/pass@1:...
```
The `[batch-safety-patch]` line firing is a GOOD sign, not a bug — it's last session's
verified fix for a known non-deterministic tokenizer edge case
(`WORKFLOW_PORT_NOTES.md` bugs 11/12) engaging correctly, not crashing.

**Expected, NOT a red flag, on this tiny untrained sanity model:**
- Reward near 0 / mixed `ENV_DONE`+`MAX_RESPONSE_LENGTH_EXCEEDED` terminations — a
  0.5B untrained model on a hard multi-hop task with a tight token budget is a wiring
  check, not a convergence test.
- `[mask-diagnostic] NON-CUMULATIVE step break` lines — a known, already-understood
  tokenizer non-determinism (`WORKFLOW_PORT_NOTES.md` bug 11/12's root cause), handled
  gracefully by the batch-safety patch above, not something to chase.

**Actual red flags** (stop and investigate, don't proceed to a real cloud run):
- A traceback anywhere in the log (`grep -iE "traceback|exception" sanity_spike.log`).
- The process exits with no `step:2` / `Final validation metrics` line at all.
- `[batch-safety-patch]` or `[mask-diagnostic]` NOT firing when you'd expect them to
  (silence isn't automatically bad, but if a masking-related assertion fires instead —
  a real crash, not just a log line — that's the actual gate to take seriously; see
  `HANDOFF.md` §4/Step 4 on why masking bugs are the dangerous, silent kind).

If all of the above pass: the environment is genuinely ready. Move on to whatever
`HANDOFF.md`'s "current status" section (or `TRAINING_HISTORY_LOG.md` /
`rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md`) says is actually next — this doc's job
ends at "environment confirmed working," not "what to train."
