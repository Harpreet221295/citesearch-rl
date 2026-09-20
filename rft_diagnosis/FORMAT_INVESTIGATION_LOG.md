# Format investigation log — 2026-08-26

**What this is.** The chronological trail of a one-session investigation into how this
agent talks to the model: what was checked, in what order, why each step was taken, what
it found, and where the reasoning went wrong along the way. The companion to
`TRAINING_HISTORY_LOG.md` (which logs training runs) for work that isn't a training run.
`RFT_PLAN_AND_MODEL_DIAGNOSIS.md`'s "Harness audit" section holds the *conclusions*; this
holds the *path*, including the parts that didn't work out.

**Why it's worth having.** The headline finding — that two sessions of results were
measuring our own code rather than the model — was not found by being clever. It was
found by one instruction ("check the format first") and by reading raw transcripts
instead of aggregates. That method is more reusable than any individual bug below, and
methods only transfer if the sequence is written down, not just the answers.

**One-line summary.** The queued next experiment was cancelled; eight bugs were found in
the native tool-calling harness and its headline result (`correct_rate = 0.0`) was shown
to be an artifact; the bracket harness was audited and has the same disease in a smaller
dose; a new open question replaced the old plan.

**Commits:** `a79b78c` → `c89232d` → `3bcec49` → `10da474` → `655c34c` (~05:34–06:39).

---

## Step 0 — the state this session inherited

`HANDOFF.md` was unambiguous about what to do next: edit the `answer` tool's schema
description in `diagnosis1_native_tools.py` to add a terseness rule, re-run the 16×4
probe, compare. The reasoning behind it was specific and plausible — the native format's
`correct_rate` was 0%, and a real sampled answer had been pulled showing the model
burying a correct fact in a sentence (`"...released in 2010, a song written Blaine
Larsen..."`) that strict exact-match couldn't credit. Verbosity, therefore, was the
problem; the fix targeted verbosity.

I staged exactly that edit. It was ready to run.

**What changed the plan:** Harpreet's instruction —

> *"forget about the threshold for diagnosis 1, at least first confirm we are using the
> right format for the model and turn boundaries n all in whatever we are trying to do"*

This is the pivot the whole session turns on. Worth stating plainly: the queued fix would
have run, produced a number, and that number would have been meaningless.

**One thing done before pivoting that turned out to matter:** the script wrote its
results to `diagnosis1_native_results.json` — the *baseline it was being compared
against*. Running it as-is would have destroyed the comparison point. Output was
redirected to a new filename. Small, but the kind of thing that silently erases evidence.

---

## Step 1 — the tell was already in the data, and had been read past four times

Before running anything, re-read the *existing* `diagnosis1_native_results.json` raw
rows rather than its summary:

```
n_parse_failures distribution: {2:4, 3:5, 4:7, 5:8, 6:4, 7:7, 8:24, 9:5}
rows with >=1 parse failure: 64/64        <- EVERY trajectory
terminated_cleanly:          10/64
made >=1 real tool call but NEVER answered: 49/64
```

The loop runs 9 rounds. A mode of **8** means nearly every round of nearly every episode
was rejected. The aggregate that summarised this was `parse_ok_rate: 0.0`, which had been
read four separate times as "noisy" rather than as what it literally says.

**Lesson worth keeping:** an extreme aggregate is a statement about the harness, not the
model. `0.0` and `1.0` deserve suspicion in a way that `0.31` does not.

**Hypothesis formed here:** in Qwen's native tool-calling convention, a model finishes by
writing ordinary text — there is no `answer` tool. Our code invented one and scored the
correct behaviour as an error. If true, one bug explains three symptoms at once (good
engagement, almost nothing finishing, 0% correct).

Deliberately *not* acted on yet. The raw text wasn't stored in the JSON, so the
hypothesis was unfalsified.

---

## Step 2 — build the instrument before drawing conclusions

`verify_format.py`, in two halves.

**Offline half (no GPU).** Render the *exact* message dicts the code builds through the
real installed tokenizer and print them verbatim, beside the canonical OpenAI-style
shape. Also dump the template source and probe the parser with six hand-written replies.

Findings:

| check | result |
|---|---|
| our `arguments` as a JSON **object** vs canonical JSON **string** | **ours is correct**; Qwen's own spec line says `{"name": ..., "arguments": <args-json-object>}`. The canonical form would have been wrong. |
| where our instructions live | in the **user** message — the real system prompt was Qwen's stock *"You are Qwen, created by Alibaba Cloud."* |
| does the parser accept a plain sentence | **PARSE FAILURE** — hypothesis survives |
| does the `search` schema declare `k` | **no**, yet the code injected it into the history |

That first row matters as much as the bugs: a thing that looked wrong was checked and
turned out right. Had it been "fixed" on intuition, a working component would have
broken.

**GPU half.** Real generations, every raw completion printed verbatim with vLLM's
`finish_reason` and `stop_reason`, failures bucketed by **cause** rather than counted.

---

## Step 3 — what the transcripts actually showed

```
plain_prose_no_tool_call   28  (77.8%)
parsed_ok                   8  (22.2%)
finish_reasons: {'stop': 36}
```

**Turn boundaries were fine.** All 36 rounds ended naturally; none truncated by the
256-token cap. A worry eliminated cleanly — worth noting that a negative result is a
result.

**Everything else was broken**, and the raw text made it undeniable:

| gold answer | model wrote | harness did |
|---|---|---|
| `RCD Mallorca` | **`RCD Mallorca`** | rejected |
| `yes` | **`yes`** | rejected |

Two of four probe questions had the exact gold answer produced and discarded. It also
produced `Christopher Reeve [Switching Channels] [Bob Holiday]` — correct citation
format — rejected.

**And a bug nobody would have predicted.** Our rejection message contained the literal
string `<tool_call>`. Fed back in a user turn, the model either halted dead at exactly
that point, or reproduced the message with a **gear emoji (U+2699)** substituted where
that string belonged, five rounds running, never recovering. Confirmed at byte level
(`M-bM-^ZM-^Y` = U+2699) rather than eyeballed.

**Third:** the model emits several tool calls per reply and the parser kept only the
first. One discarded remainder contained `answer("reporter")` — a complete answer. Same
bug class as the `_ACTION_RE` fix of 2026-08-25, never applied to this path, and
partly our fault: Qwen's own preamble says *"You may call one or more functions"* and
nothing stopped generation at `</tool_call>`.

Verdict: the plan doc's "native format: 0% correct" was dead as a claim about the model.

---

## Step 4 — a correction from Harpreet, mid-investigation

> *"all the generation n stuff, please use vllm for that, try not to use sequential or
> HF generation please, can't afford to be slow"*

Checked every call site rather than answering from memory. vLLM everywhere, no HF
`generate()` — but **two batch-1 loops**, one of them in `verify_format.py`, which I had
written *after* the instruction. 4 questions × 9 rounds = 36 engine calls that should
have been 9.

Rewritten to lockstep batching. Recorded here because the failure mode is instructive:
using the right library is not the same as using it the right way, and "it's only a
diagnostic" is how the wrong pattern gets copied into the thing that matters.

---

## Step 5 — rebuild, and re-measure

`native_rollout.py`: batched lockstep (one `generate()` per round over all active
episodes), plain text accepted as the final answer, retry message with no poisoning
literal, `</tool_call>` stop paired with `include_stop_str_in_output=True` (the two only
work together — without the second, the closing tag is stripped and every call looks
truncated), a real system message, `k` declared.

| | old | fixed |
|---|---|---|
| replies readable | 0% | **100%** |
| episodes finishing | 16% | **100%** |
| correct | 0% | **4.7%** |

Also, per Harpreet's request, human-readable transcripts. **The first version was
unreadable** — it dumped the full prompt every round, but the prompt *grows* each round,
so a 9-round episode repeated everything nine times across 40KB. Rewritten to print the
opening prompt once and then only what's new. Logged because the fix isn't obvious in
advance and the first attempt genuinely obscured the thing it was built to reveal.

---

## Step 6 — "is the model even outputting any thought?"

Harpreet's question. Measured instead of guessed: across every reply containing a tool
call, text before the call appeared **0 out of 8** times. Plain-text answers ran 2–5
tokens. **No reasoning at all.**

Chasing the cause found three, all pointing the same way:

1. Qwen's tool preamble never mentions reasoning.
2. Our instructions never asked for it.
3. **Our instructions forbade it** — `_INSTRUCTIONS` said *"No sentence, no
   explanation."* Aimed at the final answer; applied to every turn.

**The uncomfortable part:** the cancelled terseness fix would have copied that exact line
*into the answer tool's schema* to push it harder — doubling down on the instruction
suppressing reasoning.

Also found **bug 7**: the assistant message carried `tool_calls` but no `content`, so any
reasoning the model *did* write was deleted from the history. Qwen's template renders it
(`{%- if message.content %}`, verified in the template source).

Adding one line asking for reasoning:

| | without | with |
|---|---|---|
| wrote any reasoning | 3% | **55%** |
| used `read` | 0.20 | **0.47** |
| citation F1 | 0.042 | **0.094** |
| 3+ hop episodes | 13/64 | **29/64** |
| **correct** | 4.7% | **4.7%** |

**Reasoning improves the process and does nothing for the outcome.** A clean
counterexample to "more multi-hop behaviour must mean more accuracy."

---

## Step 7 — the missing 2×2 cell

The bracket arm always carried three worked examples; the native arm never did. So
bracket's correctness lead could not be attributed to *syntax* — it might just be
*demonstrations*. `native_examples.py` translates the same three trajectories into native
conversation turns: same questions, same hop counts, same search results, same passage
text, same reasoning sentences, same answers. Only the syntax differs.

**Result: examples made it worse.** Never-used-a-tool went **1.6% → 11% → 42%** as
examples and then reasoning were added. The model imitates the examples' *answers*
instead of their *process*.

**A hypothesis I formed and then killed.** I proposed that bracket's lead came from
answering out of parametric memory without retrieving — plausible, since 56% of its runs
never touched a tool. Checked it: **0 of its 12 correct trajectories had zero tool
calls.** All twelve genuinely retrieved. Recorded so it isn't re-formed.

That left the real question sharper and less comfortable: bracket ~43% correct when it
engages, native 1.8–8.1% however configured, and **not** explained by examples or
reasoning, since native now has both.

---

## Step 8 — audit the other arm before trusting its number

Deliberate discipline: 42.9% was about to decide which format the whole RFT collection
pass uses, and it came from a harness carrying the *same* warning signature
(`parse_ok_rate: 0.000`, 56% zero-tool-call) that started this investigation.

**First obstacle, itself a finding.** The bracket transcripts came out saying
`(nothing captured)`. Production `env.py` **drops the model's raw text on a parse
failure** — `_run_tool` receives it as `raw` but stores `Step(call=None, ...)`
(env.py:122). 257 rejected replies had left no record of what the model said. That is
*why* this hid so long.

Worked around by holding the env objects and reading `env._history`, rather than patching
`env.py` — adding the field would change `traj.tool_calls` counting and silently move
every existing number mid-investigation. Flagged as its own future change.

**Findings:**

- **Q1, does it reason in bracket format: yes.** 45–50% of episodes write a `Thought:`,
  ~170–190 chars. vs 3% native. ReAct gets reasoning free because the format has a slot.
  A genuine format property — and a real confound in the original A/B, which compared
  syntax while the arms also differed on whether reasoning happened at all.
- **Q2: same bug, confirmed from raw text.** `round 0 RAW: 'no'` rejected, then sometimes
  our own error message parroted back.

**Magnitudes, which cut against how I first framed it:**

- Error-echo is **minor** here — 7/384 replies (1.8%), 2/64 episodes — not dominant as in
  the native arm. I over-weighted this initially and corrected it.
- Rescue: 4 of 31 zero-tool-call episodes were already exactly right and discarded;
  14.1% → 20.3%. One was `no [Atrocious (film)] [Gebo And The Shadow]` — correct **and**
  cited, binned for lacking `answer[...]`.
- **Asymmetry on culpability:** a bare `no` genuinely violates the bracket contract (the
  prompt demands `answer[...]`); natively it is the correct way to finish. Same symptom,
  different fault. Worth not flattening into "the same bug."

---

## Step 9 — a correction to my own reporting

Three runs, identical config, temperature 0.9:

```
correct_rate:  0.188   0.156   0.141
```

A ~5pp swing from sampling alone at n=64. I had been quoting "18.8% vs 4.7%" as though
the decimals carried information. **They don't at this sample size.** The
bracket-vs-native gap survives the noise; nothing smaller stated today does.

**Standing rule from this:** at n=64 do not chase differences under ~5pp. Raise n or
sweep seeds first.

---

## Things I got wrong this session

Kept deliberately — a log that only records the wins teaches the wrong method.

1. **Batch-1 generation in `verify_format.py`**, written *after* being told to batch.
2. **The first transcript format** was so verbose it buried its own evidence.
3. **Over-weighted the error-echo** in the bracket arm before measuring it (1.8%, not
   dominant).
4. **The parametric-memory hypothesis** — plausible, wrong, killed by data.
5. **Quoted noisy numbers as precise** until three runs of the same config forced the
   correction.
6. **Walked into a documented trap.** Six background watchers hung forever on
   `until ! pgrep -f "bash setup_pod.sh"` — the watcher's own command line contains the
   pattern, so `pgrep` matched itself. `FRESH_POD_SETUP_AND_SANITY_CHECK.md` documents
   exactly this for `pkill -f`, and I read that doc at the start of this session. Now
   filed as bug #5b, with the note that reading the doc was not sufficient to avoid it.
7. **A transcript footer that contradicted the metrics** — it compared the raw answer
   string, so `'RCD Mallorca [RCD Mallorca]'` printed `exact match: no` while the scorer
   correctly counted it right. Fixed to use the same `strip_citations` + `exact_match`
   path. A transcript that disagrees with the metrics is worse than none.

---

## Where it left the project

**Settled:** the native harness's 0% was ours. Turn boundaries are fine. Our argument
encoding is right. Examples hurt the native format. Reasoning improves process not
outcome. ReAct reasons free, native calling does not. Bracket's lead is not memory-
guessing. The 2-turn collapse survives every fix (50/64 one-search-then-answer).

**Open, and honestly so:** why bracket-when-engaged (~43%) beats native-when-engaged
(~5–8%) by 6–9×. Not examples, not reasoning.

**Cheapest next cuts:** per-question overlap between the arms (if they succeed on the
same questions, difficulty dominates and the format gap is smaller than it looks); read
bracket's 12 winning transcripts against native's 3; get sample sizes past the noise
floor before deciding anything.

**Not done, and it should be said:** `python train_dr.py sanity` — the mandatory
end-to-end GRPO check from `FRESH_POD_SETUP_AND_SANITY_CHECK.md` §5 — was never run this
session. The rest of the chain passed (38 tests, imports, CUDA, dry-run). The training
path is therefore **unverified on this pod**; nothing measured here depended on it, but
it is the first thing to run before any training.

**Perspective worth keeping:** finding these bugs was necessary but does not advance the
capstone. No model was trained this session. The deliverable is still a trained agent
with an honest eval.
