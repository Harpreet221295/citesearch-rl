# START HERE — deep_research_agent

**Last updated: 2026-09-08 (end of the RL-from-SFT session; `RL_FROM_SFT_LOG.md` is the full trail).**

> **2026-09-08: RL HAS BEEN RUN on the SFT checkpoint, once** (`deep_research_agent_rl_from_sft_correct_only`,
> 99 steps). Dev: correct-and-properly-cited 0.484 → 0.570. **Held-out (n=300): 0.380 → 0.393 — inside
> noise**; the process transferred (read-before-cite +3.5, capped −17%), correctness did not measurably.
> Read [`RL_FROM_SFT_LOG.md`](RL_FROM_SFT_LOG.md) first — design, the SFT A/B (F19), **F18** (a
> training-signal bug in every earlier GRPO run), **F23** (held-out is hard-only), and `HANDOFF.md`'s
> new top section for what to do next.

**This file is the entry point. Read it first.** Then read `HANDOFF.md`'s **top section
only** (`## ⚡ CURRENT HANDOFF — 2026-08-26`) for the step-by-step commands.

| file | what it is |
|---|---|
| **`START_HERE.md`** (this) | status · the findings register · navigation · what to check before RL |
| **`HANDOFF.md`**, top section | the baton — exact commands, in order, for the next run |
| `HANDOFF.md`, everything below that | **history** from 2026-08-24/25/26-early. Four older "START HERE" sections, each superseded by the one above it. Useful for reasoning-at-the-time; **not instructions.** |

---

## Status in five lines

1. **SFT: three adapters exist.** `deep-research-agent-sft` (418 ex, 2026-08-26, held-out
   c&c 30.7%), **`…-sft-correct-only`** (1,209 ex, 2026-09-07 — the RL start; held-out c&c
   38.0%, dev 48.4%), `…-sft-imitate-all` (2,682 ex incl. wrong answers; commits more, cites less).
2. **RL from `correct_only`: run once, 99 steps** (W&B `t2n91x71`). Dev c&c 0.484 → 0.570;
   **held-out 0.380 → 0.393 (inside noise)**; read-before-cite 0.860 → 0.895, capped 0.117 → 0.097.
   **MuSiQue (OOD, n=300): correct 0.220 → 0.260 and capped 0.530 → 0.443, both significant** —
   the process change generalises. Best checkpoint (step 75) on the Hub. `RL_FROM_SFT_LOG.md` §4.33–4.36.
3. **F18**: every August GRPO run trained on trajectories truncated to 256 tokens with their
   reward zeroed. Fixed. Their conclusions are reinterpreted (F1 + F18 + F22).
4. **F23**: `heldout_eval`'s HotpotQA is 100% hard-level; dev/rl_train are 16–18% hard. Every
   dev-vs-held-out gap in this lab carries that offset.
5. `train_dr.py sanity` passes on this pod; the finish-line defects (#16/#17) are fixed but
   not yet exercised — dry-run them before the next long run.

---

## Do this first, in order

```bash
# 1. environment (fresh pod, ~15 min; see FRESH_POD_SETUP_AND_SANITY_CHECK.md)
bash setup_pod.sh
source .venv-deep-research/bin/activate
python -m pytest -q tests/            # expect 45 passed
python -m pytest -q distill/tests/    # expect 18 passed

# 2. THE CHECK THAT HAS NEVER RUN — do not skip, RL depends on it
python train_dr.py sanity

# 3. pull the SFT adapter you will start RL from
hf download harpreet22happy/deep-research-agent-sft --local-dir ./sft_adapter
```

---

## The findings register — everything learned, with status

`CONFIRMED` = measured and holds. `KILLED` = hypothesis tested and disproved.
`OPEN` = genuinely unresolved.

| # | finding | status | where |
|---|---|---|---|
| **F1** | **A citation only scores if the agent actually `read` that passage** (`tp = cited & gold & read`). Every healthy GRPO run converged on search→answer and never read — so a citation was **unscoreable by construction**. Four reward designs were tuning the incentive on an action that never happened. **This reinterprets Attempts 1–4 and Probes 1–4.** | CONFIRMED | `TRAINING_HISTORY_LOG.md` (SFT entry), `distill/DATA_COLLECTION_LOG.md` §2.2 |
| **F2** | The native tool-calling harness had **8 bugs**; its reported `correct_rate = 0.0` was measuring our code, not the model. Two gold answers were produced and discarded in a 4-question probe. | CONFIRMED | `rft_diagnosis/FORMAT_INVESTIGATION_LOG.md` |
| **F3** | The **bracket harness has the same disease, smaller dose** — 4/64 correct answers discarded for lacking `answer[...]`. Its error-echo is minor (1.8%), unlike native's. | CONFIRMED | `rft_diagnosis/FORMAT_INVESTIGATION_LOG.md` §8 |
| **F4** | **Worked examples HURT the native format.** Never-used-a-tool goes 1.6% → 11% → 42% as examples then reasoning are added. The model copies the examples' *answers*, not their *process*. | CONFIRMED | `rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md` "Harness audit" |
| **F5** | **Reasoning improves process, not outcome.** `--think` doubles reads (0.20→0.47) and citation-F1 (0.042→0.094), moves 3+-hop episodes 13→29 of 64, and leaves `correct_rate` at **exactly 0.047**. | CONFIRMED | same |
| **F6** | **ReAct gets reasoning for free (45–50% unprompted); native tool-calling does not (3%).** A real confound in the original format A/B, which compared syntax while the arms also differed on whether reasoning happened. | CONFIRMED | same |
| **F7** | Bracket's lead is **NOT** parametric-memory guessing — 0 of its 12 correct trajectories had zero tool calls. *(hypothesis I formed and killed with data)* | KILLED | same |
| **F8** | **Run-to-run noise is ~5pp at n=64** (0.188 / 0.156 / 0.141 on identical config). Do not treat sub-2-point differences as real. | CONFIRMED | same |
| **F9** | **SFT installs read-before-cite**: 0.003 → 0.810 held-out, and correct-and-cited 0.0% → 30.7%. Beats prompting (0.198) by 4x. | CONFIRMED | `distill/RESULTS.md` §1 |
| **F10** | **Most of the raw *correctness* gain is available from prompting alone** (0.030 → 0.193); SFT adds 0.193 → 0.417 on top. The distinctive SFT win is read-before-cite, not answering. **Do not report the correctness delta as a general capability win.** | CONFIRMED | `distill/RESULTS.md` §7 |
| **F11** | **The 2-turn collapse is gone.** Tool calls peak at 5 (search→read→search→read→answer); searches 2.21 vs reads 2.28, balanced. The *prompted* model still peaks at 2–3, the historical pattern. | CONFIRMED | `distill/RESULTS.md` §4 |
| **F12** | **16% of SFT episodes never terminate** — they search until the budget runs out. Held-out: 0.064 correct, only 19% produce any answer (dev: 0.000 / 6%). Stuck episodes **search 2.4x** more than successful ones. A failure to *commit*, not to read. Cause: trained only on trajectories where the teacher SUCCEEDED, so it never saw concluding under uncertainty. | CONFIRMED | `distill/RESULTS.md` §3 |
| **F13** | Excluding stuck episodes the model is **0.482 correct / 0.817 cite_f1** — the headline understates it. | CONFIRMED | `distill/RESULTS.md` §3 |
| **F14** | **`distill/SFT_RL_PLAN.md` §3 is WRONG.** It says tier-B trajectories should skip the final answer turn; given F12 that teaches process while never demonstrating conclusion — the exact broken habit. Re-test with the answer turn graded. | CONFIRMED (plan defect) | `distill/SFT_RL_PLAN.md` banner, `distill/RESULTS.md` §3 |
| **F15** | **~40 GiB of GPU memory is unexplained** in the SFT trainer. Both hypotheses tested and neither accounts for it: checkpointing on/off is 45.8 vs 46.2 GiB, and Liger fused CE recovered only 5 GiB. Current setting is safe; this is upside. | **OPEN** | `distill/BATCH_SIZE_TUNING.md` §5 |
| **F17** | **Teacher collection is COMPLETE**: all 4,000 `sft_collect` questions, 1 trajectory each, 100% inside that split with ZERO leakage into sft_dev/rl_train/heldout (verified against real task_ids). 1,210 pass the strict gate — **2.9x the 418 the shipped adapter trained on**, so a stronger starting policy for RL is available for free. | CONFIRMED | `distill/DATA_COLLECTION_LOG.md` |
| **F16** | **7 of 16 probe questions were solved by nobody**, GPT-4.1-mini included. Some fraction of this task is beyond this corpus/model size — a real ceiling on what RL can reach. | CONFIRMED | `rft_diagnosis/FORMAT_INVESTIGATION_LOG.md` |
| **F18** | **`data.max_response_length` bounds the MERGED multi-turn response, not one turn.** rLLM merges a trajectory into one row `[action0, obs1, action1, …, answer]`, right-truncates it to that key, and writes the reward into the tensor ONLY if the un-truncated response fits. We set it to the per-turn 256 from the first scaffold, so every trajectory longer than 256 merged tokens was trained on its first turn only AND scored 0. **Every GRPO run in TRAINING_HISTORY_LOG.md (Attempts 1–4, Probes 1–4) was affected; it mechanically explains the 2-turn collapse** (only short trajectories ever got paid). `response_length/clip_ratio` was the fraction of rewards thrown away. Fixed: `cfg.verl_max_response_length=4096` + per-turn `max_tokens` passed explicitly. Reinterprets F1/F11's history yet again. | CONFIRMED (2026-09-07) | `RL_FROM_SFT_LOG.md` §4.18–4.19 |
| **F19** | **Distillation-vs-rejection A/B on the SFT data.** `sft_correct_only` (1,209 correct-and-perfectly-cited episodes) vs `sft_imitate_all` (2,682 process-clean episodes, 558 wrong answers graded). imitate_all commits more (capped 13.3% → 10.2%, correct 0.625 → 0.656) but cites less completely (correct-and-properly-cited 48.4% → 33.6%) because "cited ⊆ read" admits the teacher's half-citing. RL starts from correct_only. Both on the Hub (`…-sft-correct-only`, `…-sft-imitate-all`). | CONFIRMED (2026-09-07) | `RL_FROM_SFT_LOG.md` §4.11–4.17 |
| **F20** | **`distill/validate.py` had never validated grounding at scale**: it reloaded a 2,048-question draw per record (and the wrong split), so lookups fell through to a soft warning. Fixed; all 4,000 teacher trajectories now re-derive byte-identically against the corpus. | CONFIRMED (2026-09-07) | `RL_FROM_SFT_LOG.md` §4.5–4.6 |
| **F21** | **`cfg.max_train_hours` is not enforced by anything** — a config field only. Every cloud run was killed by hand. Set the budget via `steps` from a measured `timing_s/step`. | CONFIRMED (2026-09-07) | `RL_FROM_SFT_LOG.md` §4.19 |
| **F23** | **`heldout_eval` is a harder distribution than dev/rl_train**: its HotpotQA half is 100% `hard`-level (validation split), vs 16% (sft_dev) and 18% (rl_train) from the train-split pool. SFT scores 0.64 dev / 0.52 held-out for this reason, not overfitting; RL's dev gain (+9 c&c) shrank to +1.3 held-out (inside noise) while process gains (read-before-cite, capped rate) carried over. | CONFIRMED (2026-09-08) | `RL_FROM_SFT_LOG.md` §4.34 |
| **F24** | **RL from SFT, first honest result:** dev +8 correct / +9 c&c at every checkpoint; held-out +0.3 / +1.3 (95% CI ±5); **MuSiQue (OOD) +4.0 correct, −8.7 capped, both significant, reads up / searches down.** RL taught "read, then commit"; the payoff is largest where the task is hardest. No collapse, no inflation. | CONFIRMED (2026-09-08) | `RL_FROM_SFT_LOG.md` §4.33–4.36 |
| **F22** | The historical runs trained with **no KL term**: `cfg.kl_coef` → `algorithm.kl_ctrl.kl_coef`, unused unless `algorithm.use_kl_in_reward=True`. `rl_from_sft` uses `actor.use_kl_loss=True`; with LoRA, verl's reference is the adapter-DISABLED weights, so the SFT adapter is MERGED into the base first (`distill/merge_sft.py`) to make the reference the SFT policy. | CONFIRMED (2026-09-07) | `RL_FROM_SFT_LOG.md` §1.1–1.2 |

---

## What to read, by what you want

| you want to… | read |
|---|---|
| run RL (the next stage) | `distill/SFT_RL_PLAN.md` §4–5, then **F12/F14 above** |
| see any number from this project | **`distill/RESULTS.md`** — every table, both eval sets, all three arms |
| understand why the old GRPO runs failed | `TRAINING_HISTORY_LOG.md`'s SFT entry (it reinterprets them), then **F1** |
| know how the SFT was done | `distill/SFT_HISTORY_LOG.md` |
| collect more teacher data | `distill/DATA_COLLECTION_LOG.md` §4 |
| debug memory / batch size | `distill/BATCH_SIZE_TUNING.md` |
| understand the harness audit | `rft_diagnosis/FORMAT_INVESTIGATION_LOG.md` |
| set up a fresh pod | `FRESH_POD_SETUP_AND_SANITY_CHECK.md` |

**Superseded — history only, do NOT follow as instructions:**
`HANDOFF.md`'s four OLDER sections (everything below its top `CURRENT HANDOFF` section) · `rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md`'s A/B table
(stamped) and its Diagnosis 2 procedure (rejection-sampling from the student's own
rollouts — dead, there was nothing to mine) · `POD_LIFECYCLE.md` (stop/resume era).

---

## Artifacts

| what | where | note |
|---|---|---|
| SFT adapter (418) | `harpreet22happy/deep-research-agent-sft` | private, 228 MB, r=32. Verified by fresh download + sha256, and loads into peft and generates. |
| SFT adapter (1,209, **RL start**) | `harpreet22happy/deep-research-agent-sft-correct-only` | 2026-09-07, r=32, card generated from the dev eval |
| SFT adapter (2,682, imitate-all) | `harpreet22happy/deep-research-agent-sft-imitate-all` | 2026-09-07, reference only |
| RL adapter (step 75) | `harpreet22happy/deep-research-agent-grpo` @ `deep_research_agent_rl_from_sft_correct_only__07_09_2026__23_59_34`, `best/step75_dev_cc0.570` (+ `checkpoint/` = step 75) | r=16 LoRA **on the merged correct-only weights** — rebuild them with `distill/merge_sft.py` before loading |
| **FULL RL checkpoint for RESUMING** (optimizer state incl.) | same repo/branch, `FULL_VERL_CHECKPOINT_FOR_RESUME__rl_from_sft_correct_only__global_step_75/` | 355 MB verl checkpoint + `run_meta.json` + `resolved_config.json` + a README with the exact resume steps. Not a standalone model — use `best/step75_…` to run it. |
| RL eval JSONs | `distill/eval_rl_sft_dev.json`, `distill/eval_rl_heldout.json`, `distill/eval_rl_musique_dev.json` | committed, per-episode rows |
| Teacher data | `harpreet22happy/deep-research-agent-trajectories` | private, **4,000 trajectories — the COMPLETE sft_collect split**, 1,210 pass the strict gate. **The JSONL is gitignored — the Hub copy is the only one that survives pod termination.** |
| Eval JSONs | `distill/eval_sft_A_*.json`, `rft_diagnosis/*.json` | committed, raw per-episode rows |
| Training curves | `distill/run_histories/` | committed |
| Transcripts | `rft_diagnosis/transcripts/` | committed — real model output, the thing that found every bug |

---

## Two things to check before the RL run (cheap, offline, high value)

**1. Is F16's ceiling real?** 7 of 16 probe questions were solved by nobody, GPT included.
That does NOT mean RL cannot crack them — teacher distillation is a STARTING POINT, not a
ceiling. RL here optimises against **ground truth** (exact-match on the gold answer), not
against teacher imitation, so the policy is not bounded by the teacher's distribution: if
exploration ever finds a correct trajectory the teacher missed, GRPO reinforces it. And
the teacher's failures were not obviously capability failures — it searched badly, or
honestly answered "unknown". A different query finds the passage.

But those 7 fail for one of three reasons and only two are fixable by RL:

| cause | can RL fix it? | how to test (offline, ~10 min) |
|---|---|---|
| BM25 cannot surface the gold passages at all | **No** — corpus ceiling | run each question's `supporting_titles` through the retriever; check recall@k, same shape as `data.selfcheck()` |
| Evidence is reachable, nobody found the query | **Yes** — this is exactly what exploration is for | as above; if recall is fine, the gap is query formulation |
| Answer is right but exact-match rejects it (`Nairn, Scotland` vs gold `Nairn`) | **Worse than no** — RL would actively PUNISH the correct answer | read the failed answers by hand against gold |

If the third case dominates, **fix the metric before training against it.** We already saw
one instance of it in the teacher data.

**2. Resolve F15 (the ~40 GiB) if an 8B run is planned.** At 3B it is an annoyance —
batch 2 works. At 8B, weights alone go 6 → ~16 GiB, and if that overhead scales at all no
batch may fit. A `torch.cuda.memory._record_memory_history()` snapshot would likely settle
it in one run. Note `NOTES.md`'s "multi-GPU is a SPEED choice, not a fit requirement" was
written for 3B and probably stops being true at 8B.

---

## Planned after RL: the same pipeline at 7-8B on 2x A100

Harpreet, 2026-08-26. Partly for the result, partly because learning agentic RL with
rLLM/veRL means learning multi-GPU training.

**What makes it a real comparison rather than two unrelated runs:** keep the splits, the
teacher data, the prompt, and `distill/eval_sft.py` IDENTICAL. All four are already frozen
in code, so the delta genuinely is model size. The 4,000 trajectories are model-agnostic —
no re-collection needed (~$9 saved).

**One decision to revisit at 8B:** the non-reasoning teacher (`gpt-4.1-mini`) was chosen
FOR a 3B student, on the reasoning that a small model imitating long chains learns to start
what it cannot finish. An 8B may be able to finish. "Does a stronger teacher help at 8B and
hurt at 3B?" is then a real experiment, and a cheap one — re-collect a slice with a
reasoning model and compare.

---

## The next RL run — config and the one risk

- Start from the **SFT adapter**, not the base model.
- `include_worked_example=False`. Training and rollout prompts **must match**.
- `rl_train` split (14,500 questions, disjoint from SFT — `splits.py` asserts it).
  `cloud_preset` uses `prompts_per_step=1`, so a 252-step run touches ~252 distinct
  questions; raise it rather than assuming the pool size does the work.
- Report against `heldout_eval`; early-stop on `sft_dev`. Reuse `distill/eval_sft.py` —
  it already does the 3-arm comparison. Do not rebuild it.

**The risk to watch from step 1:** the outcome reward pays for answering, so the cheapest
way to stop getting zeros on F12's stuck episodes is to **answer immediately** — which
collapses straight back to the 2-turn pattern and undoes read-before-cite. Same shape as
Probe 3, where citation attempts collapsed once avoiding them became safer.

**Watch turns-per-episode and read-rate in the first ~20 steps.** If turns fall toward 2,
it is hacking the outcome reward; strengthen the citation gate before spending hours.
Probe 4's `cite_gated` design is the counter, and unlike then, **the model can now
actually cite** — that combination has never been tested.
