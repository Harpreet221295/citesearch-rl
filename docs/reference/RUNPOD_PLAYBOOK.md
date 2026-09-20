# RunPod / Cloud Lab Playbook

The reusable **operations manual** for running any lab or capstone on a rented GPU
(RunPod single-GPU; most of it applies to Modal too). `COMPUTE.md` tells you *which
tier* a task needs and *why* (memory/throughput); **this file tells you how to build
a lab so a multi-hour cloud run is cheap, watchable, and crash-proof.**

If you're a Claude Code session building a new lab: **replicate these patterns.** The
canonical, battle-tested implementation is [`assignments/finqa_agent`](./assignments/finqa_agent)
— copy its `train.py` / `config.py` / `setup_pod.sh` / `RUNPOD.md` / `HANDOFF.md` and
adapt. Every pattern below says *why*, the config knob, and the file to crib from.

> Guiding philosophy (from CLAUDE.md + COMPUTE.md): **develop local, rent to run, tear
> down.** The local 4060 run is a *proof-of-life* that de-risks cloud spend — it is NOT
> the deliverable. Get the pipeline green + the dataset verified locally, then rent an
> A100 only for the real run.

---

## Frameworks: rLLM + verl (for the agentic capstones)

The agentic labs (finqa_agent migration, deep_research) move off the hand-rolled HF harness
onto **rLLM (agent layer) + verl (engine layer)**. Read
[`VERL_RLLM_PRIMER.md`](./VERL_RLLM_PRIMER.md) first — it explains, from first principles:
what verl is, colocated vs disaggregated weight-sync, single- vs multi-GPU rollout+training,
the pluggable backends (FSDP/FSDP2/Megatron × vLLM/SGLang), the rLLM `Agent`/`Environment`/
`reward_fn`/`AgentTrainer` API, and the **finqa → verl config Rosetta table**.

⚠️ **Before trusting any rLLM/verl syntax or config key, web-search the *current* docs — and
version-check.** These frameworks move fast and make **breaking changes across minor
versions**; the primer's snippets are accurate as of the version noted in it, not forever.
The **single biggest source of friction on a rented pod is version matching** (rLLM ↔ verl ↔
vLLM/SGLang ↔ torch ↔ transformers/peft pins) — a mismatch is what burns pod hours, not
memory or algorithm bugs. So:

- **Pin every framework version** in `requirements.txt` and record them in `pip-freeze.txt`
  (as finqa_agent does), so a run is reproducible and a fresh pod rebuilds identically.
- **`WebSearch`/`WebFetch` the docs for *your installed version*** when writing config —
  especially the unit convention for `ppo_mini_batch_size`, backend flag names, and the
  agent/env API, all of which have shifted across releases.
- **Verify the install path green on a tiny run first** (a stubbed env + `math_tool`) before
  porting real tools/reward — proves the version stack is compatible before you spend on the
  real run. This is the framework analogue of the local proof-of-life above.
- Canonical verl docs: <https://verl.readthedocs.io/> · rLLM docs: <https://docs.rllm-project.com/>
  (also mirrored at <https://rllm-project.readthedocs.io/>).
- **rLLM's own doc pages are unreliable for exact pins — three different pages gave three
  different answers this session** (including three different `AgentTrainer` call shapes).
  Verify against the raw `pyproject.toml` / source file at the exact commit you're pinning,
  not a rendered doc summary.
- **A working rLLM+verl matrix was resolved 2026-08-23** (`rllm@main` + `verl==0.9.0` override
  + `torch==2.11.0` + `vllm==0.22.1`) — three real install bugs found and fixed (flash-attn's
  build-isolation/torch chicken-egg problem, a numpy<2/numpy>=2 conflict between rLLM's stale
  `verl==0.8.0` pin and vllm, and a hatchling build-dep casualty of `--no-build-isolation`),
  plus a CUDA-version-mismatch red herring that cost real time before the actual bug (transient
  build contention) was found. Full blow-by-blow + the exact working install commands:
  [`assignments/deep_research_agent/RLLM_VERL_INSTALL_NOTES.md`](assignments/deep_research_agent/RLLM_VERL_INSTALL_NOTES.md).
  **Check this before re-deriving the matrix for a new lab** (e.g. the still-unfinished
  `finqa_agent/verl` migration) — versions will drift, but the failure MODES (isolation,
  stale cross-repo pins, doc unreliability) are likely to recur.

---

## The per-lab cloud checklist

A lab is "cloud-ready" when it has all of these. Tick them off:

- [ ] **`config.cloud_preset()`** — one preset that scales the knobs for the A100 (bigger model, `G`, batch), and turns on the safeguards below.
- [ ] **Secrets in `.env`** (`HF_TOKEN`, `WANDB_API_KEY`), gitignored, `load_dotenv()` at the top of `train.py`/`evaluate.py`.
- [ ] **`.gitignore`** covers `.venv-*/`, `runs/`, `wandb/`, cached datasets, `.env`.
- [ ] **Batched generation** (RL rollouts, AND any judge/verifier model) — never batch-1 on cloud.
- [ ] **Rollout generation runs in `model.eval()`**, not `.train()`, if the loss step uses
  gradient checkpointing — dropout + checkpointing's forced no-cache corrupts generation.
- [ ] **`--max-hours`** wall-clock cap (bounds spend).
- [ ] **Checkpoint/resume** — full state, every N steps + on the time cap.
- [ ] **Checkpoint mirrored to HF** — survive a full pod wipe.
- [ ] **`--push-to-hub`** — the trained adapter lands on HF, not just pod disk.
- [ ] **W&B** default for cloud — scalars + sample table + histograms + a periodic held-out eval curve.
- [ ] **Dataset verified locally** — actually load the real dataset before renting.
- [ ] **`setup_pod.sh`** + **`RUNPOD.md`** + **`HANDOFF.md`** for the pod session.
- [ ] **Timing-probe-first** discipline documented.
- [ ] **Decide the persistence plan BEFORE deploying, not after** — one pod session
  (nothing survives even a stop unless you finish in one sitting), or multiple sessions
  over days (build everything on Volume Disk, `/workspace`, from the start — see "Storage:
  Container Disk vs. Volume Disk vs. Network Volume" below). Cheaper to decide up front
  than to migrate mid-project.

---

## Storage: Container Disk vs. Volume Disk vs. Network Volume

Learned the hard way on `deep_research_agent` (2026-08-24) — worth getting right before
deploying rather than migrating mid-project like that session had to. Three separate
storage classes on a RunPod pod, each with a different lifetime (verified against
[docs.runpod.io](https://docs.runpod.io/storage/network-volumes), not assumed):

| storage | typical mount | survives a **stop**? | survives **terminate**? |
|---|---|---|---|
| Container Disk | `/` | NO — wiped | NO |
| Volume Disk | `/workspace` | **YES** | NO |
| Network Volume | (opt-in, attach at deploy time) | YES | **YES** |

**The mistake to avoid:** cloning the repo and building the venv under `/root/...` or
anywhere on `/` by default (which is what a fresh SSH session's `$HOME` usually is) — that
lives on Container Disk, so it's gone on the very next STOP, not just a terminate. If the
plan is a single pod session start-to-finish, this doesn't matter. If the plan is multiple
sessions over days (stop between sessions to pause billing, start again to resume — this
is normal and cheap: RunPod bills a stopped pod's Volume Disk at DOUBLE the per-GB rate
but with compute at $0, so idle cost drops by ~99% while everything on `/workspace`
survives), **build the repo + venv on `/workspace` from the very first `setup_pod.sh` run**,
not `/root`. Note a Network Volume specifically must be selected AT pod deployment — it
"cannot be attached or detached later without deleting the Pod" per RunPod's own docs — so
Volume Disk (already provisioned by default on most templates) is the practical default for
"survive a stop," and a Network Volume is only worth the extra opt-in step if the plan is
to survive a full terminate/redeploy too (e.g. reusing a built venv across genuinely
different pod rentals, not just pausing the same one).

**A `setup_pod.sh` idempotency trap this surfaced:** installing a tool (`uv`, in this
case) manually mid-session, outside the script, and never adding it to the script itself.
The script LOOKS complete and works fine — until it's re-run from a truly fresh venv
(e.g. after a storage migration, or a different pod), where it dies immediately on
`command not found` under `set -euo pipefail`. If you `pip install`/`apt-get install`
something ad hoc while debugging, add it to `setup_pod.sh` in the same sitting — the
script is only as reproducible as its worst untested rerun.

**Idle pods are still the real money sink for compute** (the line below is about the GPU
hour rate, not storage) — but "idle" and "stopped" aren't the same thing: a genuinely
idle-but-RUNNING pod bleeds the full ~$1.5-2/hr GPU rate; a STOPPED pod (with the repo on
Volume Disk) costs pennies/day and comes back ready to go. Stop between sessions, don't
leave it running; terminate only once nothing more will be pulled off that pod (checkpoints
should already be on HF by then regardless, per the checkpoint-mirroring pattern below).

---

## The patterns (why · knob · crib-from)

### 1. Secrets via `.env` (never hardcode, never commit)
`HF_TOKEN` (model/dataset download + Hub push) and `WANDB_API_KEY` go in a gitignored
`.env`. `train.py`/`evaluate.py` call `load_dotenv()` at import. `setup_pod.sh` prompts
for them and appends to `.env`.
*Gotcha:* the W&B **API key is 40 chars** (from `wandb.ai/authorize`) — a 36-char UUID is
a *key ID*, not the key. HF whoami: `HfApi().whoami(token=...)['name']`.

### 2. `.gitignore` essentials
```
.venv-*/
runs/
wandb/
data/<big-cached-dataset>   # e.g. finqa_*.json (~80MB)
.env
__pycache__/
```
Never commit venvs, run outputs, W&B dirs, multi-MB datasets, or secrets.

### 3. Batched generation — THE cost lever (RL labs)
On-policy RL generates `G` rollouts per prompt every step. **Generating them one at a
time (batch-1) wastes ~90% of a datacenter GPU** — HF `generate` gets its speed from
batching. Batch-1 turned a ~$15 run into a ~$250 one in finqa_agent.
- **Do:** generate all P·G rollouts concurrently (left-padded), chunked by a
  `gen_batch_size` knob to bound KV-cache VRAM. For multi-turn agents, recover each
  row's exact tokens by truncating at its own eos, and keep the loss-mask exact per row.
- **Verify** the batched path reproduces the single-sequence path **bit-for-bit** with a
  test (so batching can't silently corrupt the mask/logprobs).
- **Crib:** `rollout.batched_rollout` / `_run_batch` + `tests/test_tool_mask.py::test_batched_matches_single`.
- vLLM is a further speedup but needs LoRA weight-sync into the engine each step — use
  rLLM/veRL for that rather than hand-rolling; HF-batched on an A100 is the verified path.
- **Generation batch ≠ loss batch.** The generation batch (`gen_batch_size`, `no_grad`,
  KV-cache-bound) is cheap; the **LOSS forward grades all survivors of a micro-batch at
  once with grad**, and its float32 logits are `(B, T, vocab)`. With a big vocab (Qwen
  ~152k) × long multi-turn sequences (~2k tok) that's **~1.2 GB *per sequence*** — so a
  large `prompts_per_step·G` OOMs even 80 GB. **Bound the loss batch** (keep
  `prompts_per_step` small, e.g. 1, and recover the effective batch via `grad_accum`);
  they're separate knobs for a reason. The timing probe (`--steps 10`) is your OOM gate.
  If bounding it still isn't enough, go a level finer: **chunk the loss batch itself**
  (`loss_chunk_size`, e.g. 4-at-a-time within a group of 16) — forward+backward+accumulate
  each sub-chunk immediately instead of holding the whole group's graph at once. Every
  chunk must normalize by the SAME global token-count denominator a full-batch call would
  use (an optional `denom` override on the loss fn), or per-chunk gradients don't sum to
  the true full-batch gradient — verify this numerically (chunked-sum == full-batch loss
  AND gradients, bit for bit) before trusting it. **Crib:** `config.loss_chunk_size`,
  `loss.grpo_loss`'s `denom` param, the chunked loop in `train.compute_micro_batch`.

### 3b. Generate in `eval()` mode, not `train()` — a DIFFERENT axis from batching
`@torch.no_grad()` around rollout generation stops autograd graphs from being built —
it does NOT stop LoRA dropout, which only depends on `model.training`. If your loss step
needs **gradient checkpointing** for memory (pattern #3 above), you have a landmine:
checkpointing forces `use_cache=False` **whenever `self.training` is True**. Generate
with the model left in `.train()` mode (dropout on) + checkpointing enabled, and every
autoregressive step recomputes the WHOLE growing sequence through a **fresh dropout
mask** — including tokens already "generated" earlier in the same rollout — so the
model's own view of its past keeps reshuffling and output degenerates into actual
gibberish. Not a masking bug, not an OOM: the generated text itself is garbage, and it's
100% reproducible, not sampling noise (verified: identical prompt, `train()`+cache-ON →
coherent, `train()`+cache-OFF (checkpointing) → gibberish).
- **Do NOT** "fix" this by dropping gradient checkpointing — that likely just brings back
  the OOM from pattern #3 (chunking alone may not be enough headroom).
- **Do** toggle modes around the two phases: `model.eval()` before rollout generation
  (checkpointing's training-only conflict never engages), `model.train()` only for the
  loss forward/backward (where checkpointing is actually needed and generation doesn't
  happen). Exploration already comes from temperature/top_p sampling, not LoRA dropout,
  so eval-mode generation loses nothing. **Verify empirically**, not just by argument —
  we were wrong once already about which layer of the stack was the actual cause.
- **Crib:** the `was_training = model.training; model.eval(); ...; model.train()` wrapper
  around `generate_group_rollouts` in `train.compute_micro_batch` (mirrors the same
  toggle `assert_shift_correct` already used, for the unrelated reason below).

### 3c. Batch EVERY model that generates, not just the policy
An LLM-as-Judge (or any auxiliary verifier model) is just as batchable as the policy
rollouts — and just as easy to forget. A judge scored one trajectory at a time in a
Python loop cost up to `group_size·prompts_per_step·grad_accum` sequential batch-1
`generate()` calls to a SECOND full-size model, invisible until the policy rollouts
themselves were fixed and the judge became the dominant remaining cost. Same fix as
pattern #3: left-pad, one `generate()` call per `gen_batch_size` chunk. Measured **44x**
at n=16 on a real judge, with bit-for-bit identical scores vs the serial path — verify
that identity, don't just trust the speedup.
- If the reward function calls the judge *internally* per-trajectory (natural for a
  from-scratch implementation), retrofit it with an optional precomputed-score
  parameter (defaults to the old per-call behavior) rather than restructuring the
  reward math itself — the same "add an override, don't rewrite the objective" move as
  the loss-chunking `denom` param above.
- **Crib:** `judge.score_batch` / `_hf_score_batch`, `reward.reward_finqa`'s
  `precomputed_judge_score` param, the batched call in `train.compute_micro_batch` and
  `evaluate.run_eval` (used by both standalone eval AND periodic in-training eval).

### 4. `--max-hours` — bound the spend
Per-step time on a new GPU/config is a *guess*. A hard wall-clock cap stops the run at N
hours (saving/pushing what's trained) so the bill can't surprise you. Pair it with a
**timing probe** (`--steps 10`) to measure per-step cost before committing.
- **Knob:** `cfg.max_train_hours` (finqa cloud default 5). **Crib:** the `for step` loop
  check + `save_checkpoint` before the break in `train.py`.

### 5. Checkpoint / resume — survive an interrupted run
Save a **full** training state, not just the adapter: `adapter + optimizer + scheduler +
step + RNG(torch/cuda/numpy/python) + wandb_id + config`. Atomic write (`.tmp` → `os.replace`).
`--resume` restores all of it and continues at `step+1` with the exact lr-schedule
position, optimizer momentum, and sampling stream.
- **Knobs:** `checkpoint_every` (finqa cloud 25), `--resume`. **Crib:** `save_checkpoint`
  / `load_checkpoint_state` / `_restore_rng` in `train.py`, `model.load_adapter_weights`
  (peft `set_peft_model_state_dict`).
- **Caveats:** resume with the **same `--steps`** (scheduler consistency). The data loader
  is **fast-forwarded** to the resume step (deterministic same-seed shuffle → the resumed
  run sees the EXACT prompt stream an uninterrupted run would have — verified), so no
  re-seeing of early prompts.

### 5b. Keep BOTH a LATEST and a BEST — they serve different purposes
Two separate artifacts, don't conflate them:
- **LATEST** (`runs/<run>/checkpoint/`, pattern #5) — the full training state (optimizer +
  scheduler + step + RNG + config), overwritten each save. **`--resume` reads this.** You
  must resume from the *latest* (it has the optimizer/RNG and is current) — never from a
  "best" (no optimizer state; would rewind training).
- **BEST** (`runs/<run>/best/step<N>_solve<S>/`) — the **top-K adapters by held-out eval**
  (adapter only), for the deliverable. On-policy RL eval commonly **peaks then degrades**
  (over-optimization / reward hacking / KL drift), so the last step is often NOT your best.
- **Knobs:** `checkpoint_every` (latest) + `keep_best_k` (best, cloud 2). **Crib:**
  `maybe_save_best` + `load_best_index` (rebuilds the best list on `--resume` so it isn't
  lost) in `train.py`; best #1 is mirrored to HF `best/`. Requires `eval_every>0`.
- **Eval gate: point `evaluate.py --adapter` at the BEST dir, not `runs/<run>/adapter`
  (the last step).** `train.py` prints the best path at the end.

### 6. Mirror the checkpoint to HF — survive a FULL pod wipe
Pod disk vanishes if the whole pod is destroyed (not just interrupted). Mirror each
checkpoint dir (adapter + `state.pt` + `config.json`) to the HF repo under `checkpoint/`
on the run branch; on `--resume`, **pull it from HF when local disk is empty**. Now
resume works on a brand-new pod.
- **Knob:** `push_checkpoints` (cloud True) + `push_to_hub_repo`. **Crib:**
  `_push_checkpoint_to_hub` (`create_repo`/`create_branch`/`upload_folder`) +
  `_pull_checkpoint_from_hub` (`snapshot_download`) in `train.py`.
- **Cost:** ~100MB/mirror — raise `checkpoint_every` if the cadence is heavy. A RunPod
  persistent volume for `runs/` is an alternative (avoids the upload).

### 7. `--push-to-hub` — the deliverable leaves the pod
Push the trained LoRA adapter to a **private** HF repo, branch = `run_name` (one revision
per run). Never commit multi-GB weights to git. **Test it early** — a missing push means
the adapter dies with the pod.
- **Crib:** `push_adapter_to_hub` in `train.py` (`model.push_to_hub(repo, private=True,
  revision=run_name)`). Verify with a real push once (`--sanity --push-to-hub <repo>`).

### 8. Weights & Biases — watch a multi-hour run from anywhere
Default for cloud (`log_backend="wandb"`). Log more than scalars:
- **Scalars:** reward, solve-rate, KL, lengths, per-tool adoption, per-difficulty, etc.
- **Sample table:** a few real rollouts (input → action trace → output vs gold) so you can
  eyeball *behavior*, not just curves.
- **Histograms:** reward / advantage distributions (catch collapse the mean hides).
- **Periodic held-out eval curve** (`eval_every`) — a real generalization signal logged
  during training so you can **early-stop from the dashboard** instead of guessing steps.
- **W&B resume:** pass `id=<saved wandb_id>, resume="allow"` so a resumed run continues the
  *same* dashboard. Store the `wandb_id` in the checkpoint.
- W&B also **auto-logs GPU util/mem** — free OOM/underutilization signal.
- **Crib:** `train.Logger` (+ `log_table`/`log_hist`), `run_periodic_eval`,
  `evaluate._log_eval_to_wandb`. *Note:* personal W&B is free; ignore the "Team trial" nag.

### 9. Verify the real dataset LOCALLY before renting
Do not trust a dataset id or assume a schema. **Download and load the real dataset on the
free box first.** finqa_agent's configured HF dataset was **unloadable** (`datasets>=3`
dropped script-based datasets) and the loadable mirror had the wrong schema — both would
have crashed on the pod at \$2/hr. We switched to the original JSON and verified fields +
counts locally.
- **Rule:** a lab's data path is "done" only after `load_tasks(...)` runs on the real data
  locally and you've eyeballed a sample + the split sizes. Cache large downloads under a
  gitignored dir.
- **Ship a `prepare_data.py`** that downloads both splits upfront and prints counts +
  sample + distribution + a leakage check, **exiting non-zero on red flags**; call it from
  `setup_pod.sh` so a dataset/network problem halts setup instead of surfacing mid-run.
  **Crib:** `assignments/finqa_agent/prepare_data.py` + `data.py`.

### 10. `setup_pod.sh` — one-command pod bootstrap
Git creds (for push-back), optional Node+Claude Code (skip if driving via Desktop SSH),
tokens → `.env`, the isolated `.venv-<lab>`, a CUDA sanity check, and a printed next-steps
list. **Crib:** `assignments/finqa_agent/setup_pod.sh`.

### 11. `HANDOFF.md` — hand the baton to the pod's Claude session
A self-contained brief so a fresh Claude Code session on the pod is productive immediately:
what the lab is, what's DONE (don't rewrite), the exact command sequence, what to watch,
the honesty gates, and the one likely first-run snag. Pair it with a paste-in opening
prompt. **Crib:** `assignments/finqa_agent/HANDOFF.md`.

### 12. Timing-probe-first + pin versions
Run `--steps 10` first, read per-step wall-clock, THEN size the run. `pip freeze >
pip-freeze.txt` and commit it (RL-for-LLM stacks are version-fragile). Pin seeds + model
revisions.

### 13. Honesty gates (from CLAUDE.md — non-negotiable)
- Never report "reward went up" as success — the deliverable is the **eval gate**
  (base-vs-tuned on held-out + localized metrics + reward-hacking probes).
- Report **unmeasured numbers as unmeasured** (timing, projected cost).
- **Update `/BUILD_LOG.md`** after the run so the tutor session can sync.

---

## The pod runbook (generic)
```bash
# on a fresh A100 80GB pod, inside the cloned repo:
cd assignments/<lab> && bash setup_pod.sh      # creds, tokens->.env, venv, CUDA check
pip freeze > pip-freeze.txt
python -m pytest -q tests/                      # mechanics green
python -c "import data, config; ..."            # VERIFY the real dataset loads
python train.py --cloud --steps 10 --run-name timing_probe   # measure per-step cost
python train.py --cloud --push-to-hub <you>/<lab>            # W&B + max-hours + checkpoints ON by default
#   ... if the pod dies: relaunch the SAME command + --resume (pulls checkpoint from HF)
python evaluate.py --adapter runs/<run>/adapter --cloud                 # the eval GATE
python evaluate.py --adapter runs/<run>/adapter --cloud --no-fewshot
# push code/eval_report.json/README; adapter is on HF; then TEAR THE POD DOWN.
```

## Gotchas we actually hit (so you don't)
- **`datasets>=3` won't load script-based HF datasets** ("Dataset scripts are no longer
  supported"). Verify locally; fall back to the source JSON/parquet.
- **W&B key vs key-id:** the 40-char string from `wandb.ai/authorize`, not a 36-char UUID.
- **Batch-1 rollouts are a silent ~10× cost bomb** on cloud. Batch them.
- **`assert_shift_correct`-style two-forward self-checks are flaky under LoRA dropout** —
  run them with `model.eval()` (dropout off); they verify indexing, not dropout.
- **Gradient checkpointing + `model.train()` + LoRA dropout = corrupted generation**, not
  just slower generation. Checkpointing forces `use_cache=False` while `self.training`,
  so every generate() step recomputes the whole sequence through a fresh dropout draw —
  the model loses a stable view of its own past and rollouts degenerate into gibberish.
  Generate in `.eval()`, only `.train()` for the loss step (pattern #3b). We initially
  misdiagnosed this as a chat-template masking bug (it looked identical from the
  symptoms: `tools=0, turns=1`) — always check whether the *generated text itself* is
  coherent before assuming a masking/bookkeeping bug.
- **Qwen3's chat template pre-seeds an empty `<think>\n\n</think>\n\n` into the
  GENERATION PROMPT itself** when priming the next turn with `enable_thinking=False` —
  and strips think-blocks from HISTORICAL turns when re-serializing. If you're tracking
  an append-only token history for a masked multi-turn RL loss (ReAct-style agents), that
  stub gets committed once (as injected/mask-0 prompt tokens) but never reappears when
  that turn becomes history — a guaranteed, 100%-reproducible append-only mismatch on the
  very next turn, not sampling noise. Strip the stub from the INJECTED prompt tokens
  before committing them (not from the model's generated continuation — the stub isn't
  something the model generates, it's already-closed template output the model just
  continues past). Same fix needed in both a single-sequence reference rollout function
  AND its batched counterpart if you keep both (they must stay behaviorally identical).
- **Adapter/checkpoint on pod disk is lost on a full wipe** — mirror to HF (pattern #6).
- **The loss forward OOMs from `(B, T, big-vocab)` float32 logits**, not model size — a
  0.5B and a 4B have the same 152k vocab. Long multi-turn sequences amplify it. Bound the
  loss batch (pattern #3); shorten `max_turns`/`max_new_tokens` if still tight.
- **Two resident models (policy + a same-size judge) want an 80 GB card** — don't squeeze a
  two-4B-model RL workload onto 48 GB to save a few $/hr; you'll fight OOM on paid time.
- **The LATEST checkpoint isn't your best model** (RL peaks then regresses) — keep top-K by
  eval and eval the BEST (pattern #5b).
- **Idle pods are the real money sink** — tear down (or at least STOP, see "Storage:
  Container Disk vs. Volume Disk vs. Network Volume" above) the instant you're done for
  the session.

---
*Reference implementation for every pattern here: [`assignments/finqa_agent`](./assignments/finqa_agent)
(see its `BUILD_LOG.md` entries for the blow-by-blow of why each decision was made).*
