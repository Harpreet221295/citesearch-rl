# Results — every number, in one place

**Date:** 2026-08-26. Generated from `eval_sft_A_dev.json`, `eval_sft_A_heldout.json`,
`runs/sft_A/history.json`, and `teacher_trajectories.jsonl`. This is the reference
appendix for the eventual write-up: narrative lives in
[`SFT_HISTORY_LOG.md`](SFT_HISTORY_LOG.md), numbers live here.

Three arms everywhere, because two would answer the wrong question:

| arm | what it is | what it tells you |
|---|---|---|
| **base (no example)** | base model, example-free prompt | what SFT started from |
| **base + worked example** | same weights, worked example in the prompt | what prompting alone achieves |
| **tuned** | SFT adapter, example-free prompt | what we built |

Comparing tuned only against *base* flatters SFT — the base model is being denied a
demonstration it used to get. The middle arm is the skeptic's control.

---

## 1. Held-out set (n=300) — THE HEADLINE NUMBERS

Locked split, never trained on, never used for any decision. Greedy decoding.

| metric | base (no example) | base + worked example | **tuned** |
|---|---|---|---|
| correct (exact match) | 0.030 | 0.193 | **0.417** |
| picked the right sources | 0.007 | 0.291 | **0.744** |
| verified them (read first) | 0.003 | 0.198 | **0.810** |
| citation-F1 (the reward) | 0.002 | 0.110 | **0.710** |
| called `read` at least once | 0.070 | 0.360 | **0.993** |
| finished cleanly | 0.537 | 0.927 | **0.873** |
| never used a tool | 0.897 | 0.107 | **0.000** |
| answer length (words) | 1.750 | 1.717 | **1.893** |
| citations per answer | 0.177 | 0.797 | **1.697** |
| tool calls per episode | 6.123 | 3.210 | **5.363** |
| reads per episode | 0.113 | 0.590 | **2.283** |
| searches per episode | 0.093 | 1.513 | **2.207** |
| **correct AND properly cited** | **0.0%** | **0.3%** | **30.7%** |

## 2. Dev set (n=96) — used for decisions, reported for completeness

| metric | base (no example) | base + worked example | **tuned** |
|---|---|---|---|
| correct (exact match) | 0.052 | 0.365 | **0.594** |
| picked the right sources | 0.008 | 0.336 | **0.731** |
| verified them (read first) | 0.000 | 0.193 | **0.807** |
| citation-F1 (the reward) | 0.000 | 0.132 | **0.714** |
| called `read` at least once | 0.052 | 0.344 | **1.000** |
| finished cleanly | 0.438 | 0.927 | **0.844** |
| never used a tool | 0.917 | 0.135 | **0.000** |
| answer length (words) | 1.208 | 1.458 | **1.521** |
| citations per answer | 0.094 | 0.917 | **1.583** |
| tool calls per episode | 6.240 | 3.083 | **5.438** |
| reads per episode | 0.115 | 0.542 | **2.312** |
| searches per episode | 0.104 | 1.458 | **2.281** |
| **correct AND properly cited** | **0.0%** | **1.0%** | **44.8%** |

---

## 3. THE BIMODAL FAILURE — the most important thing in this document

The tuned model has two modes, and averaging them hides both.

**Dev set, n=96** (headline correct_rate 0.594)

| | n | correct | cite_f1 | produced an answer | reads | searches |
|---|---|---|---|---|---|---|
| finished under the cap | 80 (83%) | 0.713 | 0.856 | 100% | 2.05 | 1.88 |
| **hit the 8-call cap** | 16 (17%) | 0.000 | 0.000 | 6% | 3.62 | 4.31 |

**Held-out set, n=300** (headline correct_rate 0.417)

| | n | correct | cite_f1 | produced an answer | reads | searches |
|---|---|---|---|---|---|---|
| finished under the cap | 253 (84%) | 0.482 | 0.817 | 100% | 2.07 | 1.81 |
| **hit the 8-call cap** | 47 (16%) | 0.064 | 0.131 | 19% | 3.45 | 4.36 |


**The capped episodes are not slow — they are near-total losses.** The model searches and
reads until the 8-call budget runs out, and usually never answers at all.

Note the two eval sets differ in severity and the difference is worth stating precisely
rather than rounding away: on **dev** the capped episodes are *complete* losses (0.000
correct, only 1 of 16 producing any answer). On **held-out** a minority do squeeze out an
answer at the buzzer — 9 of 47 (19%), giving 0.064 correct and 0.131 cite_f1. Either way
they are ~7x worse than the episodes that finish (0.482 correct), so the characterisation
holds; but "they always return None" is true of dev, not of held-out.

### Why it happens

We trained only on **tier A** — trajectories where the teacher *succeeded*. Every one of
those ends with a confident answer. The model never saw what to do when it *cannot* find
the answer, so it has no learned "conclude with what I have" behaviour. It keeps searching
until it runs out of budget.

**That is a classic distillation artifact: train only on successes and you teach the happy
path but never the fallback.**

### Diagnosis — it is a failure to COMMIT, not to read

Harpreet's hypothesis was *"it could be a case model is unnecessarily reading"* — partly
right, and the data refines it. On the held-out set:

| | gold passages needed | reads done | searches done |
|---|---|---|---|
| finished normally | 2.25 | 2.07 (0.92x) | **1.81** |
| hit the cap | 2.26 | 3.45 (**1.53x**) | **4.36 (2.4x)** |

Over-reading is real (1.53x what the question needs) but **over-searching is the bigger
excess (2.4x)**. The pattern is repeated query reformulation without ever committing. A
few capped episodes did zero or one read at all — pure search loops.

### Two things follow

**1. The headline numbers UNDERSTATE the model.** Among episodes that finish it is
**0.713 correct with 0.856 citation quality** on dev
(0.482 / 0.817 held-out). The headline is dragged down by episodes that produce no output at all.

**2. This is precisely what RL is for.** GRPO's outcome reward gives exactly zero for never
answering and something for answering. The model already produces the right shape ~83-84%
of the time, and **sharpening a behaviour that already exists is the case RL is actually
good at** — unlike Attempts 1-4 and Probes 1-4, where it was being asked to invent one.

### It reverses an earlier plan decision

`SFT_RL_PLAN.md` §3 says tier-B trajectories should grade the process turns and **skip the
final answer turn** (reasoning: don't train a confident wrong fact with citations attached).

**Given this failure that is probably wrong.** It teaches the process while never
demonstrating conclusion — the exact habit that is broken. Worth testing tier B **with the
answer turn included** despite the wrong answers, or a mixed variant.

Not fixed now, deliberately: it is a genuine finding about what SFT-on-successes does, and
the next session has the right tool for it.

---

## 4. Tool-call distribution — the 2-turn collapse is gone

Held-out set, n=300.

| tool calls | base | base + example | tuned |
|---|---|---|---|
| 0 | 127 (42%) | 6 (2%) | **0 (0%)** |
| 1 | 147 (49%) | 27 (9%) | **0 (0%)** |
| 2 | 15 (5%) | 93 (31%) | **0 (0%)** |
| 3 | 5 (2%) | 96 (32%) | **15 (5%)** |
| 4 | 4 (1%) | 37 (12%) | **65 (22%)** |
| 5 | 0 (0%) | 16 (5%) | **129 (43%)** |
| 6 | 0 (0%) | 10 (3%) | **25 (8%)** |
| 7 | 1 (0%) | 0 (0%) | **19 (6%)** |
| 8 | 1 (0%) | 15 (5%) | **47 (16%)** |

Every healthy run in `TRAINING_HISTORY_LOG.md` converged on ~2 turns — one search, then
answer, never a read. The **prompted** model still shows exactly that (peaks at 2-3). The
**tuned** model peaks at 5 calls = search -> read -> search -> read -> answer, with searches
2.21 and reads 2.28 — near-perfectly balanced, i.e. it reads what it finds
rather than searching repeatedly.

Reads per episode across the arms: **0.11 -> 0.59 -> 2.28**.

---

## 5. Outcome buckets

| bucket | base | base + example | tuned |
|---|---|---|---|
| wrong_answer | 0.970 | 0.807 | **0.583** |
| correct_uncited | 0.017 | 0.037 | **0.000** |
| correct_miscited | 0.013 | 0.153 | **0.110** |
| correct_and_cited | 0.000 | 0.003 | **0.307** |

`correct_and_cited` — correct answer AND every gold passage cited AND read — went
**0.0% -> 0.3% -> 30.7%**. That bucket was 0 in every configuration measured
across the whole day: 320 base rollouts, five format variants, and the prompted baseline.

---

## 6. Anti-hacking probes

A model can lift citation-F1 by pasting more citations, or lift nothing while inflating
length. Held-out set:

| probe | base | base + example | tuned | reading |
|---|---|---|---|---|
| answer length (words) | 1.75 | 1.72 | 1.89 | no length inflation |
| citations per answer | 0.18 | 0.80 | 1.70 | rose WITH source accuracy (0.744), so real citing, not Probe 4's paste-one-citation exploit |
| tool calls | 6.12 | 3.21 | 5.36 | more work done, matched by more reads |

---

## 7. Gate (pre-registered in code before any adapter existed)

All five PASSED on both dev and held-out:

1. read_before_cite improves >= 0.10
2. calls_read_rate improves >= 0.10
3. correct_rate does not regress
4. terminated_cleanly >= base
5. no citation-count inflation unless title_f1 also rose

**Did fine-tuning beat just prompting?** (reported, deliberately NOT a gate condition —
SFT is needed for RL regardless, since RL needs the short prompt and a policy that behaves
without the crutch)

| | tuned | prompted | |
|---|---|---|---|
| read_before_cite (held-out) | **0.810** | 0.198 | YES, 4x |
| correct_rate (held-out) | **0.417** | 0.193 | YES |

**But read this honestly:** most of the raw *correctness* gain is available from prompting
alone (0.030 -> 0.193); fine-tuning adds 0.193 -> 0.417 on top. The adapter's
*distinctive* contribution is read-before-cite and the correct-and-cited bucket. Without
the middle arm this would have been reported as a general capability win, which it is not.

---

## 8. Training run

418 tier-A trajectories, all distinct questions. LoRA r=32/alpha=64 on all attention + MLP
projections (59.9M params, 1.9%). lr 1e-4 cosine, batch 2 x grad-accum 8 (effective 16),
3 epochs, ~7 minutes on one A100.

| epoch | train loss | val loss |
|---|---|---|
| 0 | nan | 0.2722 |
| 1 | nan | 0.2565 |
| 2 | nan | 0.2563 |

Train loss falls steadily; **val flattens after epoch 1** (0.2565 -> 0.2563).
Mild memorisation — 2 epochs is likely enough at this dataset size.

Overfit sanity before the real run: **0.679 -> 0.0066 PASS** on 16 examples.

---

## 9. Teacher data

| | |
|---|---|
| episodes collected | **4,000 — the COMPLETE sft_collect split** |
| distinct questions | 4,000 (zero duplicates; k=1) |
| correct | 0.762 |
| citation-F1 | 0.667 |
| read-before-cite | 0.789 |
| **strict gate (correct AND perfectly cited)** | **1,210 (30%)** |
| API give-ups (excluded) | 1 |
| cost | $8.88 |

Buckets: `{'correct_miscited': 1344, 'correct_and_cited': 917, 'wrong_answer': 709, 'correct_uncited': 13}`

The shipped adapter trained on **418** of these — the strict-gate subset available when the
run started. **1,210 are now available (2.9x more)**, so retraining is free upside for the
RL stage's starting policy.

---

## 10. Noise floor — do not over-read small differences

Three identical-config runs earlier the same day gave `correct_rate` **0.188 / 0.156 /
0.141** at n=64. At n=300 the interval is tighter, but differences under ~2 points should
not be treated as real without more samples or a seed sweep.
