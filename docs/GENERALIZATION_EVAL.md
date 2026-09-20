# Generalisation eval — MuSiQue as the out-of-distribution set

**Date:** 2026-09-08. **Status:** loader + split + eval wiring built and tested offline;
first numbers land after the RL run finishes (the GPU is busy training). Harpreet's
question that started this: *"are there any other datasets on which we can do
generalisation testing?"*

## 1. Why MuSiQue, and why nothing else yet

Everything in this lab trains and reports on HotpotQA + 2WikiMultiHopQA (the
`heldout_eval` split is a frozen 500 from their validation sets). A model that only ever
sees those two can overfit their question templates ("what nationality is the director
of…", "which was released first…") and their 10-passage corpora. A third dataset, never
trained or collected on, tells whether the search→read→cite behaviour is a skill or a
memorised template.

| candidate | ships passages? | verdict |
|---|---|---|
| **MuSiQue** (Trivedi et al. 2022) | yes — 20 per question with `is_supporting` flags, plus `answer_aliases` | **used.** Same shape as our data, harder by construction (2–4 hops composed to defeat shortcuts), and the standard third member of the HotpotQA / 2Wiki / MuSiQue trio that Search-R1-family papers report on |
| Bamboogle (125 q) | no | needs the never-wired `wiki_index` corpus backend |
| NQ / TriviaQA / PopQA | no | single-hop, need a retrieval corpus |
| StrategyQA | yes (evidence paragraph ids) | yes/no answers, more format work; a possible fourth set |

Two cheaper generalisation cuts need no new data and are now printed by `eval_rl.py`:
per-source (HotpotQA vs 2Wiki) and per-hop-count slices of any eval, so a gain can be
attributed rather than averaged.

## 2. What was built

| file | change |
|---|---|
| `data.py` | `MUSIQUE_HF_ID = "dgslibisey/MuSiQue"`; `musique_row_to_task(row, task_id)` (pure, testable) and `_load_musique(split, n, seed)` |
| `splits.py` | `get_split("musique_dev")` → 300 questions from MuSiQue's **validation** split, drawn with `eval_seed`; `MUSIQUE_DEV_SIZE = 300` to mirror the held-out report size |
| `distill/eval_rl.py`, `distill/eval_sft.py` | `--split musique_dev`; `eval_rl.py` also reports `by_dataset` and `by_hops` slices |
| `tests/test_musique.py` | the row→task mapping incl. aliases, gold flags, hop parsing, duplicate-title disambiguation |

Nothing in the env, tools, reward or rollout code changed — that is the point of the
per-question-corpus design: a new dataset is a new loader, nothing else.

### Mapping details that would bite silently
- **Duplicate titles.** Within one MuSiQue question two paragraphs often share a title
  (two passages from the same article). `DocStore` is keyed by title and `read[<title>]`
  must resolve to one passage, so repeats become `Title (2)`, `Title (3)` — applied to the
  supporting list identically, so gold membership and `read`-before-cite still line up.
  **151 of the 300 dev questions** have at least one such repeat.
- **Aliases.** MuSiQue provides `answer_aliases` (82 of 300 questions have some);
  HotpotQA/2Wiki provide none. `DRTask.answers` includes them, so EM is a little more
  lenient here than on the in-distribution sets — the direction is *conservative* for
  the RL-vs-SFT comparison (both arms get the same leniency), but MuSiQue absolute
  numbers are not strictly comparable to `heldout_eval`'s.
- **Hop count** is parsed from the id prefix (`2hop`, `3hop1`, `4hop2`, …) into
  `meta.n_hops`; HotpotQA is treated as 2-hop for the per-hop slice, 2Wiki is unknown.
- Only the `answerable` subset is on this mirror (2,417 validation rows, all
  `answerable=True`), so there is no unanswerable case — same as our other sets.

## 3. What the set looks like (measured on the 300, 2026-09-08)

| | `musique_dev` | `heldout_eval` (HotpotQA/2Wiki) |
|---|---|---|
| questions | 300 | 300 reported (500 exist) |
| hops | 2: 141 · 3: 98 · 4: 61 | mostly 2 |
| passages per question | 20.0 | 10.0 |
| gold passages per question | 2.73 | ~2.2 |
| mean answer length (words) | 2.8 | ~1.9 |
| **BM25 recall@3 of the gold titles from the raw question alone** | **0.437** (all gold in top-3: 9%; none: 14%) | **0.705** (all gold: 43%) |

The last row is the important one for interpretation: with 20 passages and 3–4 hops,
one search from the question surfaces less than half the evidence. A policy that does
search → read → **search again with what it learned** → read → answer should separate
from one that searches once. That is exactly the behaviour SFT installed and RL is
sharpening, so MuSiQue tests the *process*, not just recall. Expect absolute numbers
well below the in-distribution ones; the RL-vs-SFT delta and the capped rate are the
quantities to read.

## 4. How to run it (after the RL run frees the GPU)

```bash
source .venv-deep-research/bin/activate
# SFT (merged weights, no adapter) vs the chosen RL checkpoint(s), same engine, greedy
python distill/eval_rl.py --split musique_dev --n 300 \
    --adapter step50=runs/deep_research_agent_rl_from_sft_correct_only/_merged/step50/lora_adapter \
    --adapter best=<path> --out distill/eval_rl_musique_dev.json
```
Same prompt, stop sequences and engine settings as every other eval; base-model arms can
be added with `distill/eval_sft.py --split musique_dev` if a from-scratch baseline is
wanted (it costs two more arms of generation).

## 5. Pre-registered expectations (written before any number exists)
- Absolute `correct` on MuSiQue for the SFT policy: well under its 0.63 dev figure —
  **[speculation]** 0.25–0.40, dominated by 3–4-hop questions.
- `capped_rate` higher than in-distribution (more hops → more chances to loop).
- If RL's in-distribution gain is real process improvement (more commits, more complete
  citation), it should carry over here at least in direction; if it is template
  memorisation, the delta should vanish. Read the `by_hops` slice: a gain confined to
  2-hop questions is the memorisation signature.
- **Do not select a checkpoint on this set.** It is reported, like `heldout_eval`, once.
