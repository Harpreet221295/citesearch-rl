# RFT plan & model diagnosis — deep_research_agent

**Date:** 2026-08-25. **Author:** Harpreet's plan, written up by Claude Code per his
instruction, on resuming this capstone on a fresh pod. **Status:** Diagnosis 1 RUN for
real against Qwen2.5-3B-Instruct (base, no adapter) — see "Diagnosis 1 — results
(2026-08-25)" below. Diagnosis 2 (RFT) not started.

**Why this doc exists:** last session's `TRAINING_HISTORY_LOG.md` ended on a real,
load-bearing finding — every healthy GRPO run converged to ~2 turns/episode (one
`search`, straight to `answer`, never `read`, never a second hop) regardless of reward
design, and the likely cause is that the system prompt never showed the model a worked
`read[...]` example or a full multi-hop trajectory. A fix (a full worked example) is
committed but was never run before the pod stopped. Rather than immediately burn cloud
GPU-hours re-launching GRPO on the fixed prompt and hoping, Harpreet's call: **diagnose
whether the base model is even capable of the target behavior at all, before spending
any RL compute on it.** GRPO can only sharpen a behavior the policy already produces
with *some* non-zero probability — it cannot teach a behavior the model never emits in
any sampled rollout. This doc is the assessment pipeline that answers that question,
built to run BEFORE the next GRPO attempt (probe or full cloud run).

Consistent with `TENTATIVE_FUTURE_EXPERIMENTS.md`'s Option B (RFT) — this doc is the
concrete plan for actually running that option, gated behind a cheaper first check.

---

## The core philosophy

Don't ask "does reward go up" first. Ask, in order:
1. **Can the model do this at all, given every possible help (rich in-context
   examples)?** — if not, no amount of RL or even naive fine-tuning on self-sampled
   data will bootstrap it, because there's nothing correct to learn from.
2. **Can the model's PARAMETERS learn the pattern**, not just imitate examples sitting
   in the prompt? — this is the actual RL prerequisite: a policy that puts non-trivial
   probability mass on multi-hop tool use *without* being spoon-fed the pattern
   every single rollout.
3. Only once (2) is true does it make sense to spend cloud GPU-hours on GRPO — at that
   point RL's job is to sharpen and correct a behavior that already exists, which is
   what RL is actually good at, rather than to discover a behavior from scratch via
   sparse-reward exploration.

---

## Data hygiene — held-out set & overfitting safeguards (2026-08-25, Harpreet's addition)

**Non-negotiable across every stage below:** carve out a frozen held-out eval set
**before** anything else runs, and never let it touch collection, filtering, or
training.

- **Three-way split, decided up front, not improvised per-stage:**
  1. **Held-out eval set** — frozen, fixed seed (reuse `NOTES.md`'s `eval_seed=9999`
     convention so it's the SAME set later GRPO eval already uses — one consistent yardstick
     across Diagnosis 1, post-RFT re-check, and any future GRPO run, not a different
     sample each time that makes results incomparable). **Never sampled for collection,
     never fine-tuned on, never used to pick the pass/fail threshold.**
  2. **Collection/training pool** — the questions Diagnosis 2 samples `k` trajectories
     from and mines correct ones out of. Disjoint from (1).
  3. **Diagnosis-1-only probe pool** — can reuse the collection pool or a separate small
     slice; Diagnosis 1 doesn't train anything, so its only hygiene requirement is not
     silently double-counting as the held-out set later.
- **Report every result (Diagnosis 1's rates, Diagnosis 2's pre/post numbers, the
  eventual GRPO eval) against the SAME held-out set.** A number that used a different
  slice each time isn't comparable — this is the same discipline `TRAINING_HISTORY_LOG.md`
  already insists on (real W&B numbers, not guessed, not re-sampled per check).
- **SFT-specific overfitting safeguards** (Diagnosis 2's fine-tune, small LoRA on a
  likely-small mined dataset — genuinely at risk of overfitting):
  - **Sanity-first, per CLAUDE.md's own standing rule**: overfit a tiny slice (~8-16
    examples) first, same shape as `assignments/sft_min`'s `train.py --sanity` — confirms
    the masking/loss wiring is correct before spending a real run on it.
  - **Track train-set metric vs. held-out-set metric every eval pass, not just train
    loss.** A shrinking train loss with a flat-or-worsening held-out multi-hop success
    rate IS the overfitting signal to watch for — don't rely on loss curves alone.
  - **Limit epochs / early-stop on held-out performance**, not a fixed epoch count
    chosen in advance — small mined datasets overfit fast.
  - **Check the mined dataset's diversity before training on it**, not just its size —
    hop-count distribution, question-type spread, and whether trajectories all echo the
    same 2-3 Diagnosis-1 worked examples too closely (citation phrasing, search-query
    style). A large-but-narrow dataset teaches a narrow behavior and will look like a
    win on the (also narrow) held-out slice if that slice happens to share the same
    narrowness — cross-check against the FULL held-out set's own diversity, not just
    raw pass rate.

---

## Diagnosis 1 — in-context capability check (cheap, no training)

**Question:** given rich worked examples, can the model follow the pattern at all?

**Build:** 2-3 hand-worked trajectory examples — real question → tool calls → tool
responses → final cited answer — with **varying length/hop-count** (e.g. one 2-hop
example, one 3-hop example with a `read` call that changes the answer, one where a
first search misses and a second search recovers). These go well beyond the current
`_opening_prompt`'s single `search`→`answer` snippet (the exact gap
`TRAINING_HISTORY_LOG.md` found) — full transcripts, not one-line syntax reminders.

**Procedure:**
1. Insert the 2-3 worked examples into the system prompt (a variant of
   `env.py::_opening_prompt`, built alongside it rather than replacing the existing
   fix — this is testing a STRONGER prompt than what's already committed).
2. Take questions from the **Diagnosis-1 probe pool** (disjoint from the frozen
   held-out eval set — see "Data hygiene" above; also run the SAME check against the
   held-out set itself at the end, as the number that will stay comparable across
   every later stage).
3. Sample **k generations per question across a SWEEP of temperatures** (e.g. `T ∈
   {0.3, 0.7, 1.0}`) — via **vLLM batched generation**, reusing `evaluate.py::
   run_batched_rollouts` (already built, already verified end-to-end on a pod last
   session) — not raw HF `.generate()`, which is what let a previous lab quietly limp
   along on unrepresentative single-sample greedy decoding. **Report results PER
   temperature, don't collapse to "the best temperature found"** — temperature serves
   two different purposes here: a high-temp sweep answers "does the capability exist
   anywhere in the distribution" (informs whether Diagnosis 2's collection pass is
   worth running at all), while the result **at GRPO's actual rollout temperature
   (~0.9, per `config.py`)** specifically is what predicts GRPO viability later — keep
   both, don't conflate them into one number.
4. For each sampled trajectory, check (reusing existing, already-tested code — no new
   verification logic needed), and **report a full breakdown, not just a pass/fail
   rate** — this is meant to be a real diagnostic, in the same spirit as
   `TRAINING_HISTORY_LOG.md`'s raw-episode-log investigations, not a single number:
   - Does it parse as valid multi-turn ReAct syntax (`env._parse_react_action`)? — log
     the parse-failure rate itself, don't just discard failures silently.
   - Does it call `read` at least once, not just `search`+`answer` (`trajectory.py`'s
     step-level action log)? Also log the hop-count distribution (1 tool call, 2, 3+).
   - Does it terminate cleanly (`TerminationReason.ENV_DONE`, not `max_turns` exhaustion)?
   - Is the final answer correct (`metrics.exact_match`) and are citations valid
     (`citations.verify_citations`, gold-membership backend)? Break down into the
     actual failure taxonomy: wrong answer / correct-but-uncited / correct-but-
     miscited / correct-and-cited — these are different diagnoses, not one number.

**Pass/fail criterion (needs Harpreet's threshold, not guessed here):** something
like — at GRPO's actual rollout temperature specifically, a *meaningful, non-trivial
fraction* of sampled trajectories per question show valid multi-hop tool use (calls
`read`, terminates cleanly) and a *non-zero* fraction reach a correct, cited answer.
The exact bar (10%? 30%?) is a judgment call to make once real numbers exist — don't
pre-commit to a number now, but do write it down once chosen, here, before looking at
results (avoid moving the goalposts after the fact).

**If this fails** (model essentially never produces a valid multi-hop trajectory even
with rich in-context help): the model may lack a more basic capability than
"citation discipline" — possibly the ReAct format itself, or multi-hop reasoning at
this model size. Fall back to **GPT-generated trajectories** (Harpreet supplies an API
key) as the seed data for Diagnosis 2 instead of self-sampled ones — see "Fallback"
below.

---

## Diagnosis 1 — results (2026-08-25, real run against Qwen2.5-3B-Instruct base)

**Setup:** probe pool = 16 questions (first slice of the train pool, `offset=0`), `k=4`
samples/question, temperatures `{0.3, 0.7, 0.9}` swept, base model (no adapter). Full
raw rows + breakdown: `diagnosis1_results_v2.json` (bracket format) and
`diagnosis1_native_results.json` (native format).

### Three real bugs found and fixed along the way (all in production `env.py`, not just
the diagnostic — see `env.py`'s own inline comments, dated 2026-08-25, for full detail)

The FIRST run (`diagnosis1_results.json`, superseded, kept out of git — see `.gitignore`)
showed `parse_ok_rate≈0.02` and 61% of trajectories with zero tool calls at GRPO's own
temperature. Investigating the raw model output (not just the aggregate numbers — same
"pull the real transcript" discipline `TRAINING_HISTORY_LOG.md` already established)
found this was NOT primarily a capability signal:

1. **`_ACTION_RE`'s greedy-regex bug.** With nothing stopping generation at the true
   turn boundary, the model sometimes free-ran past its first `Action:` and wrote an
   entire HALLUCINATED continuation (fake tool results, a fake second turn, a fake final
   answer) in one completion — imitating the worked examples' full-trajectory SHAPE
   rather than stopping after one step. The old regex (`re.DOTALL`, greedy `(.*)`,
   anchored to `$`) didn't just fail on this, it silently ACCEPTED it: verified directly
   that a `search[...]` call's `query` argument ended up containing the model's entire
   hallucinated rest-of-trajectory, including a fake final answer. When the hallucination
   got cut off mid-way by `max_new_tokens` instead, the same regex failed to match at
   all (no trailing `]` at the truncated string's end) — the OTHER failure mode, a
   spurious parse failure for a turn that actually started fine. Fixed: `re.MULTILINE`
   instead of `re.DOTALL`, confining the match to one line — `re.search` now finds only
   the FIRST complete `Action: tool[...]` line and ignores anything hallucinated after
   it. Verified NOT to regress the legitimate multi-bracket-citation case
   (`answer[American [Title1] [Title2]]`) — see `tests/test_mechanics.py`'s two new
   regression tests.
2. **No stop-sequence.** Nothing told vLLM to stop generating at the turn boundary, so
   the model could burn its whole `max_new_tokens` budget hallucinating a continuation
   the parser (even fixed) would just discard. Fixed: `stop=["\nThought:",
   "\nsearch results:", "\n["]` in `diagnosis1.py`'s `SamplingParams`.
3. **The prompt never actually said "stop after one action."** It showed the full
   multi-turn SHAPE (necessary — the model needs to see what a real observation looks
   like) but never explicitly instructed the model that ITS job is only the one
   Thought/Action step, not the whole shape. Fixed: added one explicit sentence to both
   `env.py`'s real prompt and `diagnosis1.py`'s rich prompt: "Write EXACTLY ONE
   Thought/Action pair, then STOP — do not write what the tool returns yourself; the
   real result will be given to you before your next turn."

**Verified directly, not assumed**, that the fix works: fed the model an accumulating
real trajectory turn-by-turn and confirmed each generation is exactly one Thought/Action
step (or a clean answer), never a hallucinated continuation — including a genuine
recovery case (a malformed round-0 attempt correctly triggered the env's recoverable
parse-error path, and the model corrected itself on the very next round, going on to
complete a full, correct, properly-cited 2-hop trajectory).

### Bracket-format results, before vs. after the three fixes (temperature=0.9, GRPO's own)

| metric | v1 (buggy) | v2 (fixed) |
|---|---|---|
| correct_rate (EM) | 0.08 | **0.19** |
| mean_cite_f1 | 0.00 | **0.05** |
| terminated_cleanly_rate | 0.36 | 0.41 |
| `correct_and_cited` bucket | 0% | **1.6%** |

Real, measured improvement from fixing real bugs — not guessed.

### The real finding: capability exists, but is bimodal, not reliable

At temp=0.9 (v2, fixed): one question got **4/4 correct with full, real multi-hop tool
use and proper citations** (search→read→search→read→answer, exactly the taught
pattern); another got 3/4. Clean, direct proof the capability is real. But roughly a
third of the 16 probed questions (5-6 of them) got a short, degenerate, zero-tool-call
response across ALL 4 samples instead of engaging the loop at all — grouped by task, not
randomly scattered across samples (`hotpot-train-49699`, `2wiki-train-26372`,
`hotpot-train-8437`, `hotpot-train-1982`, `2wiki-train-138326` all showed 4/4 samples
with 0 tool calls). This does NOT cleanly correlate with question type — 3 of the
collapsed questions are yes/no-shaped, but so is one of the FULLY-successful questions
(`2wiki-train-147113`, 4/4 correct) — so no overclaimed cause here; genuinely open.

**This is not "the model can't do multi-hop" (disproven by the 4/4 case) nor "the model
reliably can" (disproven by the ~1/3 collapse rate) — it's exactly the profile that
makes rejection-sampling RFT sound**: real correct-and-cited trajectories already exist
in the sampled distribution, worth mining and using to make the good behavior reliable
instead of occasional.

### A/B: does Qwen's OWN native tool-calling format do better? (same probe pool, temp=0.9)

> **⚠️ SUPERSEDED 2026-08-26 — every number in this subsection is a HARNESS ARTIFACT.**
> The native-format harness had six real bugs; its `correct_rate=0.0` measured our code,
> not the model. Read "Harness audit — 2026-08-26" at the end of this document instead.
> This subsection is kept unedited as the record of what was believed and why, because
> the reasoning that produced it (and the four separate readings of the same data that
> missed the bugs) is itself the lesson. **Do not act on the "next thing to try" below —
> it was chasing a symptom that only existed in the biased 10/64 sample that survived a
> broken code path.**

Qwen2.5-Instruct has a real, instruction-tuned-in tool-calling format (Hermes-style
`<tool_call>{"name":...,"arguments":...}</tool_call>`, verified directly against the
INSTALLED tokenizer's `chat_template` — not assumed from a web doc, which turned out to
describe Qwen3's slightly different convention when checked). Built
`diagnosis1_native_tools.py` to test it directly: proper JSON-schema tool definitions
passed via `tools=` in `apply_chat_template`, real `role="tool"` responses, NO
hand-crafted worked examples (native calling is supposed to need little/no in-context
teaching).

| metric (n=64) | bracket format | native format |
|---|---|---|
| zero-tool-call rate | 56% (36/64) | **8%** (5/64) |
| calls_read_rate | 11% | **41%** |
| terminated_cleanly_rate | **41%** | 16% |
| correct_rate (strict EM) | **19%** | 0% |
| mean_cite_f1 | **0.05** | 0.01 |

**Neither format is a clean win — they fail in different, complementary ways.** Native
format essentially solves the "gives up entirely" collapse (huge jump in engagement:
41% read-rate vs 11%, only 5/64 zero-tool-call vs 36/64) — the model reliably uses its
own trained format and explores more thoroughly. But `correct_rate` dropped to 0%.
Investigated directly (pulled real `final_answer` text, not just the metric) — this is
NOT a knowledge gap: one sample's answer was `"Gotta Get to You [I Gotta Get to You]
released in 2010, a song written Blaine Larsen..."` — the correct year (2010, matching
gold) is right there, embedded in a full restated sentence instead of the terse phrase
strict EM requires. The bracket format's problem was *engagement*; the native format's
problem is *answer discipline* (the terseness/citation instructions live in the
system-prompt text, not the `answer` tool's own JSON-schema description, and the model
seems to attend to the tool schema more than the surrounding prose when in native-calling
mode).

**A concrete, cheap next thing to try** (not yet built): native tool-calling format +
move the terseness/citation instructions INTO the `answer` tool's schema description
itself, rather than only the system prompt. This targets the specific, now-understood
failure mode rather than guessing at a bigger redesign.

### Where this leaves the plan

Diagnosis 1's core question — can the model do this at all — is answered: **yes**, with
real, direct proof (multiple 4/4 and 3/4 clean trajectories).

> **⚠️ The rest of this paragraph is SUPERSEDED (2026-08-26).** Step (a) — the
> native-format + schema-description terseness fix — was built and then NOT run, because
> auditing the harness first showed the failure it targeted was an artifact. See
> "Harness audit — 2026-08-26" below for what replaced it.

The open question is
reliability, and the native-format probe suggests it's tractable (engagement is
fixable; answer-discipline is a narrower, more targeted problem than "the model can't
reason multi-hop"). Reasonable next steps, in rough cheapest-first order: (a) the
native-format + schema-description fix above, cheap to test; (b) proceed to Diagnosis 2
(RFT) using whichever format variant shows the best correct+cited yield during a pilot
collection pass, per the plan's existing pilot-first discipline.

---

## Diagnosis 2 — RFT: can the model's parameters learn the pattern?

**Question:** after fine-tuning on the model's own correct trajectories, does it
reproduce the pattern **without** the in-context crutch?

**Procedure:**
0. **Pilot the yield first, before committing to the full collection pass.** Run the
   collection step (below) on a small slice (~50 questions × `k=8`) from the
   collection/training pool and measure the real hit-rate (what fraction of sampled
   trajectories pass the filter in step 2). Scale `k` / question count from THIS
   number, not a guess — a low hit-rate means a much bigger `k` or question count is
   needed to reach a usable dataset size, and it's cheaper to learn that from 50
   questions than from the full pass.
1. **Collect.** For many training-set questions (from the collection/training pool —
   never the held-out eval set), sample `k` trajectories/question via vLLM batched
   generation (same harness as Diagnosis 1, likely reusing the SAME rich-example
   prompt from Diagnosis 1 to maximize the yield of correct trajectories worth mining
   — this is rejection sampling, we want a good hit rate, not a representative sample
   of the un-helped policy).
2. **Filter — dual gate, both required, not just correctness.** Keep only
   trajectories that are BOTH correct (EM) AND well-cited (citation-F1 above a real
   bar, not just "attempted a citation") — reuse the exact same reward/verification
   code as training (`reward.py`, `citations.py`, `metrics.py`) rather than inventing
   a new correctness check, so "correct" means the same thing here as it will during
   GRPO later. **Filtering on correctness alone risks teaching "answer right" without
   "cite well"** — precisely the failure mode GRPO already fell into (Probes 1-4);
   RFT must not reproduce it by accident via a loose filter.
3. **Check the mined dataset's diversity before building anything from it** — hop-count
   distribution, question-type spread, and whether kept trajectories are
   suspiciously homogeneous in citation phrasing/search-query style (an echo of the
   2-3 Diagnosis-1 worked examples rather than genuine variety). A narrow-but-large
   mined set risks teaching a narrow behavior that still looks like a win on a
   similarly-narrow eval slice — see "Data hygiene" above.
4. **Build an SFT dataset** from the kept trajectories: prompt → full multi-turn
   trajectory (tool calls, tool responses, final cited answer), formatted the same way
   `env.py` constructs the ReAct conversation — but **with the in-context worked
   examples REMOVED from the prompt this time**. This is the actual test: can the
   weights internalize the pattern, or was Diagnosis 1's success purely an
   in-context-imitation artifact that evaporates the moment the crutch is removed?
5. **Fine-tune** (LoRA, matching this lab's existing compute profile — see
   `HANDOFF.md` §5), following the overfitting safeguards in "Data hygiene" above:
   sanity-overfit a tiny slice first, track held-out (not just train) metrics every
   eval pass, limit epochs / early-stop on held-out performance. Masked SFT loss over
   the trajectory (mask tool-response/observation tokens exactly the same way this
   lab's masking discipline already works — `env.py`'s `assert_verl_masking_matches`
   and `WORKFLOW_PORT_NOTES.md`'s "masking is automatic" discovery describe the SAME
   prefix-delta masking principle this SFT stage needs; `assignments/sft_min` is this
   repo's existing reference implementation of a hand-written masked SFT loss and is
   the pattern to follow, not reinvent).
6. **Re-evaluate on the frozen held-out set** the same way as Diagnosis 1 (sample k
   generations/question across the same temperature sweep, **without** in-context
   examples this time) — does it now produce multi-hop trajectories that call `read`,
   terminate cleanly, and answer correctly, on its own? Report the SAME full
   breakdown as Diagnosis 1 (parse-failure rate, hop-count distribution, the
   wrong/uncited/miscited/correct-and-cited taxonomy) so the before/after comparison
   is apples-to-apples, not just a headline pass rate. Specifically watch for the
   **collapse pattern from last session's Attempt 2** (`lr=1e-4`:
   `response_length/clip_ratio` pinned at 96-98%, model never terminates cleanly, hits
   `max_new_tokens` almost every rollout) — if RFT reproduces that shape, the fix is
   not just "give it examples," something about the fine-tuning itself needs
   revisiting (LR, data quality, overfitting to a narrow trajectory shape).

**Pass criterion:** on the frozen held-out set, without in-context examples, the
RFT'd model reliably (not just occasionally) produces valid multi-hop trajectories,
AND the held-out breakdown doesn't show a diversity/overfitting collapse (e.g. only
succeeding on the exact hop-count/question-type the mined data happened to be rich
in) — this is the real prerequisite for GRPO to have something to sharpen.

---

## Fallback — if Diagnosis 1 fails

If the base model can't follow even rich in-context examples: generate trajectories
with an external stronger model (GPT, via an API key Harpreet will supply) instead of
self-sampling. Use those GPT-generated trajectories as the seed dataset for Diagnosis
2's fine-tuning step directly (skip self-sampling/rejection-filtering, since there's
nothing correct to mine from the base model's own rollouts). Everything else in
Diagnosis 2 (masked SFT, re-evaluate without ICL examples, watch for collapse) stays
the same.

---

## Decision gate — what happens after

- **Diagnosis 2 succeeds** (RFT'd model shows real multi-hop tool use on its own,
  without ICL examples, on sampled generations): proceed to the agentic GRPO training
  from last session's plan (`cite_gated` reward + the fixed multi-hop prompt) — but
  **starting from the RFT checkpoint, not the raw base model**. This directly matches
  `TRAINING_HISTORY_LOG.md`'s own synthesis: "teach the citation mechanic first,
  cheaply, via supervised/preference methods, THEN let RL focus on the
  harder exploration-requiring parts on top of a policy that already knows the format."
- **Diagnosis 2 fails even with the GPT-trajectory fallback:** genuinely open — would
  need a fresh look at model size, task difficulty, or corpus/tooling design before
  trying RL again. Not planned for in this doc; revisit if it happens.

---

## What's new infra vs. reused (for scoping the build)

**Reused, already built and verified last session — do not rebuild:**
- `evaluate.py::run_batched_rollouts` (batched vLLM generation, LoRA-aware)
- `reward.py` / `citations.py` / `metrics.py` (correctness + citation verification)
- `data.py` (question loaders)
- `env.py`'s ReAct parsing/masking discipline

**New, needs building:**
- ~~The Diagnosis 1 sampling+scoring script~~ **DONE + RUN 2026-08-25** —
  `diagnosis1.py` (temperature sweep via `evaluate.run_batched_rollouts` reused
  unchanged, per-trajectory classification against the same `citations.py`/
  `metrics.py` scoring training uses) + `tests/test_diagnosis1.py` (13 offline
  tests) + `tests/test_mechanics.py`'s 2 new regression tests for the `_ACTION_RE`
  bug found while running this. Real results in "Diagnosis 1 — results" above.
- ~~The rich multi-example prompt variant~~ **DONE 2026-08-25** — `diagnosis1.
  rich_opening_prompt` (3 worked examples: the existing 2-hop case, a NEW 3-hop
  case where reading the full passage corrects what the snippet alone implied, a
  NEW search-miss-then-reformulate case, each followed by REAL tool-response text
  matching `tools.py`'s exact format), injected via a scoped monkey-patch of
  `env._opening_prompt` (`patched_rich_prompt`). env.py's REAL prompt also got 2
  fixes this session (the `_ACTION_RE` bug + an explicit "stop after one action"
  instruction) — see "Diagnosis 1 — results" above for why.
- **NEW, not originally planned** — `diagnosis1_native_tools.py`: a direct A/B
  testing Qwen2.5-Instruct's own native Hermes-style tool-calling format against
  the custom bracket format. Built after the bracket-format bug hunt raised the
  question of whether a custom, barely-demonstrated format was itself part of the
  reliability problem. Real, if mixed, results — see "A/B" subsection above.
- The trajectory → SFT-dataset builder (format conversion + masking) — Diagnosis 2, not built yet.
- **The SFT training loop + masked loss itself** — this lab has no SFT harness yet
  (only GRPO via `train_dr.py`). `assignments/sft_min` is the reference pattern.
  **Full-build mode for this project as of 2026-08-25 (Harpreet's explicit call) — write
  this end-to-end, not a `TODO(harpreet)` stub.** CLAUDE.md's default COACH mode is
  overridden for `deep_research_agent` going forward; see `BUILD_LOG.md` for the note.
- The "re-evaluate without ICL examples" pass (mostly a rerun of the Diagnosis 1
  harness against the fine-tuned checkpoint instead of the base model).

---

## Open decisions, not resolved by this doc (Harpreet's calls)

- Exact pass/fail thresholds for Diagnosis 1 and 2 (see the "needs Harpreet's
  threshold" notes above) — pick before looking at results, not after.
- Exact size of the held-out eval set and the collection/training pool split (the
  THREE-way split itself is now settled — see "Data hygiene" — but not yet the exact
  counts; the Diagnosis 2 pilot step will inform how large the collection pool needs
  to be for a usable mined-dataset size).
- The exact citation-F1 bar for Diagnosis 2's dual-gate filter (step 2) — "well-cited"
  needs a real number, not just "better than nothing."
- Whether the rich in-context prompt from Diagnosis 1 is ALSO worth keeping for the
  eventual GRPO stage (as a permanently richer few-shot prompt) independent of RFT, or
  whether RFT is meant to make it removable again.

---

## Harness audit — 2026-08-26 (supersedes the A/B subsection above)

> **The chronological version of this section is
> [`FORMAT_INVESTIGATION_LOG.md`](FORMAT_INVESTIGATION_LOG.md)** — what was checked in
> what order and why, including the dead ends, the hypothesis that was formed and then
> killed by data, and the six mistakes made along the way. This section holds the
> conclusions; that one holds the path.

**Trigger, and the rule worth keeping:** Harpreet's call on resuming — *"at least first
confirm we are using the right format for the model and turn boundaries, in whatever we
are trying to do"* — before running the queued terseness fix. That instinct was correct
and saved the next experiment from being meaningless. The tell was already sitting in
`diagnosis1_native_results.json` and had been read past four times: `parse_ok_rate: 0.0`
with a parse-failure mode of 8 out of 9 rounds is not "a bit noisy", it is **every round
of nearly every episode being rejected**. An aggregate that extreme is a statement about
the harness, never about the model.

**Method** (`rft_diagnosis/verify_format.py`, `audit_bracket.py`): render the real chat
template and read it; capture every raw completion verbatim with vLLM's `finish_reason`
and `stop_reason`; bucket failures by CAUSE rather than counting them. Transcripts are
committed under `rft_diagnosis/transcripts/` — one file per question, prompt shown once
then the conversation. Committing them is deliberate: last session's raw episode logs
were never saved and died with the pod, taking several findings' evidence with them.

### Eight bugs in the native harness (all measured, all fixed in `native_rollout.py`)

1. **Plain-text answers were rejected.** In Qwen's native convention a plain-text turn IS
   how the model finishes; there is no `answer` tool. We invented one and scored the
   correct behaviour as `error: no valid tool_call found`. 28/36 probe rounds. On 2 of 4
   probe questions the model emitted the EXACT gold answer (`RCD Mallorca`, `yes`) and it
   was discarded; once `Christopher Reeve [Switching Channels] [Bob Holiday]` — right
   citation format — also discarded.
2. **The rejection message contained the literal string `<tool_call>`**, which poisoned
   the conversation when echoed into a user turn: the model reproduced it with a gear
   emoji (U+2699) substituted for 5 straight rounds, or halted at exactly that point.
3. **Multiple tool calls per reply, all but the first silently binned** — one reply held
   search+read+read+`answer("reporter")`+a fake user turn. No `</tool_call>` stop, and
   Qwen's own preamble invites it ("You may call one or more functions").
4. Instructions rode in the user message; the real system prompt was Qwen's stock persona.
5. We fed back a `k` argument our own schema did not declare.
6. Generation was batch-1 in a Python loop (also true of `verify_format.py`'s first
   version, since fixed) instead of lockstep-batched.
7. The assistant message carried `tool_calls` but no `content`, deleting any reasoning
   the model wrote before a call. Qwen's template does render it (`{%- if message.content %}`).
8. Nothing ever asked for reasoning — and `_INSTRUCTIONS` actively said *"No sentence, no
   explanation."* That line targets the final answer but governs every turn.

**Checked and CLEARED, so it stays checked:** turn boundaries are fine. All 36 probe
rounds ended naturally (`finish_reason='stop'`); none hit `max_new_tokens=256`. And our
`tool_calls` arguments render as a JSON **object**, matching Qwen's own spec line — the
OpenAI-canonical JSON-**string** form renders wrong here. Verified, deliberately not
"fixed".

### The bracket arm has the same disease, in a smaller dose

Audited identically rather than assumed clean (`audit_bracket.py`). Same root cause,
confirmed from raw text: `round 0 RAW: 'no'` rejected, then sometimes our own error
message parroted back. Step causes over 64 episodes: `prose_no_react_syntax` 257,
`parsed_ok` 84, `action_line_present_but_rejected` 10.

But the magnitudes differ and should not be equated:
- Error-echo is **minor** here — 7/384 replies (1.8%), 2/64 episodes — versus dominant
  in the native arm.
- Counterfactual rescue: 4 of 31 zero-tool-call episodes were already exactly right and
  discarded; 14.1% would become 20.3%. One was `no [Atrocious (film)] [Gebo And The
  Shadow]` — correct AND cited, binned for lacking `answer[...]`.
- **Asymmetry that matters for culpability:** a bare `no` genuinely violates the bracket
  contract (the prompt demands `answer[...]`), whereas natively it is the correct way to
  finish. Same symptom, different fault.

**An observability gap in production `env.py` that explains why this hid:** on a parse
failure, `_run_tool` receives the model's raw text as `raw` but stores
`Step(call=None, ...)` — dropping it (env.py:122). A rejected reply leaves no record of
what the model said. The audit works around it by holding the env objects and reading
`env._history`; fixing it properly in `env.py` is worth doing but would change
`traj.tool_calls` counting and silently move every existing number, so it was not done
mid-investigation.

### Re-measured results (n=64 each, temp 0.9, Qwen2.5-3B base)

| arm | used a tool | correct WHEN it used one | never used a tool | writes reasoning |
|---|---|---|---|---|
| bracket + 3 examples | 28/64 | **12/28 = 42.9%** | 56% | **45-50%** |
| native (plain) | 63/64 | 3/63 = 4.8% | 1.6% | 3% |
| native + think | 63/64 | 3/63 = 4.8% | 1.6% | 55% |
| native + examples | 57/64 | 1/57 = 1.8% | 11% | 75% |
| native + examples + think | 37/64 | 3/37 = 8.1% | 42% | 58% |

**Read this table with the noise in mind.** Identical config, three runs: `correct_rate`
0.188 / 0.156 / 0.141 — a ~5pp swing at n=64. Single-decimal comparisons between arms
are not meaningful here. The bracket-vs-native gap (~14-19% vs ~4.7%) survives it;
nothing smaller does. **Any future arm comparison at this scale needs more samples or a
seed sweep before it means anything.**

### What actually holds

1. **The native harness's 0% was ours, not the model's.** Dead as a claim.
2. **Worked examples HURT the native format**, they do not rescue it: never-used-a-tool
   goes 1.6% -> 11% -> 42% as examples and then think are added. The model pattern-matches
   the examples' ANSWERS instead of their PROCESS.
3. **Reasoning improves process, not outcome.** `--think` doubles reads (0.20 -> 0.47) and
   citation-F1 (0.042 -> 0.094), moves 3+-hop episodes 13 -> 29 of 64, and leaves
   `correct_rate` at exactly 0.047. Do not assume more multi-hop behaviour buys accuracy.
4. **ReAct gets reasoning for free; native calling does not.** 45-50% unprompted vs 3%.
   That is a genuine property of the formats and a real confound in the ORIGINAL A/B,
   which compared syntax while the arms also differed on whether reasoning happened.
5. **A hypothesis formed and then killed by data**, recorded so it is not re-formed:
   bracket's lead is NOT parametric-memory guessing. 0 of its 12 correct trajectories
   had zero tool calls; all 12 genuinely retrieved.
6. **The 2-turn collapse survives every fix.** 50/64 native episodes are one search then
   answer. `--think` moves it to 34/64. Now cleanly measurable rather than buried.

**Still open, genuinely:** why bracket-when-engaged (~43%) beats native-when-engaged
(~5-8%) by 6-9x. It is NOT worked examples and NOT reasoning — the native arm now has
both. Something else differs that these runs do not isolate. **Do not design Diagnosis
2's collection pass around either format until this is understood**; picking the wrong
one determines what every mined trajectory looks like.

---

## Epilogue — 2026-08-26 (late): Diagnosis 2 ran, via the fallback

This document's Diagnosis 2 (rejection-sample RFT from the STUDENT's own rollouts) was
never viable: across 320 student rollouts, **zero** trajectories were both correct and
correctly cited. There was nothing to mine. The "Fallback" section above — seed from a
stronger model — is what actually ran.

**Result: shipped.** `harpreet22happy/deep-research-agent-sft`. Held-out (300 q):
correct-and-properly-cited **0.3% -> 30.7%**, read-before-cite **0.198 -> 0.810** (against
the base model *with* a worked example, not just the bare base).

**And the reason RFT-from-self was impossible is the same reason Probes 1-4 failed:**
`verify_citations` scores a citation only if the passage was actually `read`
(`tp = cited & gold & read`). Every healthy run converged on search->answer and never read,
so no citation could score. Four reward designs tuned the incentive on an action that never
happened.

Full narrative: [`../distill/SFT_HISTORY_LOG.md`](../distill/SFT_HISTORY_LOG.md) and
[`../distill/DATA_COLLECTION_LOG.md`](../distill/DATA_COLLECTION_LOG.md). Next stage plan:
[`../distill/SFT_RL_PLAN.md`](../distill/SFT_RL_PLAN.md).
