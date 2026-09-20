# deep_research_agent — search → read → synthesize agent (Capstone 6, Branch B)

> **New here, or resuming?** Read [`START_HERE.md`](START_HERE.md) first — current status, what to run, and a register of every finding with its status.


> Phase 4, Book §10.7. Train a small agent to answer **multi-hop** questions by
> *searching* an offline corpus, *reading* the retrieved passages, and *synthesizing*
> a grounded, cited answer — trained with **GRPO/RLVR** on the **rLLM (agent) + veRL
> (engine)** stack, mirroring `finqa_agent`. Capstone brief:
> [`../../projects/06-deep-research-agent/README.md`](../README.md).

## Goal — what I'm training and why (one line)
Teach a small Instruct model to *do research well* — issue good queries, read the right
passages, and answer multi-hop questions **grounded in what it retrieved** — via GRPO,
with the **retrieved (tool-returned) tokens loss-masked** (the L4 silent bug), and prove
a real, localized gain over a no-training ReAct baseline (not judge-gaming).

## Scope (from the capstone brief — Branch B, one config)
- **Method/stack:** GRPO (RLVR outcome reward), rLLM+veRL. **Model:** Qwen2.5-3B-Instruct
  on a single A100 (0.5B stand-in locally). **Env:** offline retrieval over a fixed corpus.
- **Data:** multi-hop QA — **HotpotQA + 2WikiMultiHopQA**, held-out split for eval.
  (SEC-10K corpus is the **stretch** — the "make it stand out" original artifact.)
- **Success bar:** beat the ReAct baseline by a clear margin on held-out multi-hop QA
  (aim for a meaningful share of Search-R1's reported ~+20% at 3B) **and** pass a
  groundedness check (answers supported by retrieved passages, no judge-gaming).

## The offline corpus trick (why local sanity is cheap)
Search-R1's real run retrieves over the **full Wikipedia** dump — heavy. But HotpotQA /
2Wiki each ship, **per question, ~10 paragraphs (2 gold supporting + 8 distractors)**.
That per-question paragraph set *is* a natural, tiny, fully-offline corpus — the exact
analogue of `finqa_agent`'s "gold table + distractor tables" design. So:
- **sanity / local:** retrieve over the **bundled per-question paragraphs** (no download,
  deterministic, BM25). Proves the loop, the mask, the reward wiring.
- **cloud (real):** point `config.corpus_backend="wiki_index"` at a full-corpus retriever
  (Search-R1's E5/BM25 Wiki index) for the number that's comparable to the paper.

`supporting_facts` (the gold paragraph titles) give us a **retrieval hit-rate** metric for
free — a localized signal the capstone asks for (don't report one number).

## What's reused vs new (the finqa → deep_research port)
```
  REUSED PATTERN (from finqa_agent)          NEW (this folder)
  ─────────────────────────────────────      ────────────────────────────
  Trajectory/Step/ToolCall + model_mask  →   trajectory.py   (search/read/answer turns)
  TableWorld + execute() tool sandbox    →   tools.py + corpus.py (search/read over a DocStore)
  data.py fixture + distractor design    →   data.py         (HotpotQA/2Wiki + bundled fixture)
  judge.py mock/hf backend               →   judge.py        (groundedness judge)
  reward.py layered reward (KNOBS in cfg)→   reward.py        (outcome+grounded+efficiency) TODO
  env.py rLLM BaseEnv wrapper            →   env.py          (DeepResearchEnv) TODO seams
  metrics.py localized eval metrics      →   metrics.py      (EM/F1, hit-rate, #steps)
  (new — no finqa analogue)              →   citations.py    (Proof-of-Use citation verify)
```
`citations.py` is the differentiator: it measures whether the answer's `[Title]`
citations point at passages the agent actually read AND that support the claim
(passage-claim alignment), flags **fabricated** citations (the sharpest tool-call-hack
tell), and counts uncited claims. Pure MEASUREMENT — the reward *policy* that folds its
`CitationReport` into a groundedness scalar is `reward.py` TODO #1.

## What's mine to write (`TODO(harpreet)` — the core learning logic)
Per COACH mode (CLAUDE.md), the plumbing is scaffolded; the credit-assignment /
reward logic is mine:
1. **`reward.reward_deep_research`** — the layered reward: outcome correctness (EM/F1
   vs gold) + **groundedness** (is the answer supported by retrieved passages? via judge
   and/or a citation-overlap check) + **efficiency** toll on extra steps. Knobs live in
   `config.py`; the *policy* that combines them is mine.
2. **`env._to_dr_trajectory`** — convert the rLLM conversation (messages + tool results)
   into the `Trajectory` the reward scores. The credit-assignment seam: decides what gets
   graded.
3. **`env.assert_verl_masking_matches`** — verify veRL's multi-turn mask grades ONLY the
   model-generated tokens, never the retrieved `<information>` passages (the L4 silent bug).
4. **GRPO advantage/loss** — owned by veRL now (we don't hand-roll it here), but I must
   read/verify its group-relative advantage + KL match what I learned in `rlvr_math`/`finqa`.

## Worklog

### Setup — env, model, dataset, framework versions, GPU
- **venv:** `.venv-deep-research` · **W&B project:** `deep_research_agent`
- **Model:** local stand-in `Qwen/Qwen2.5-0.5B-Instruct`; real `Qwen2.5-3B-Instruct` + LoRA on RunPod.
- **Data:** HotpotQA (`hotpot_qa`, distractor config) + 2WikiMultiHopQA; bundled offline
  fixture for sanity (no network).
- **Versions (fill after install):** torch ___ · transformers ___ · rllm ___ · verl ___ · vllm ___ · py ___
- **GPU:** RTX 4060 (8 GB) sanity only; single A100 (RunPod) for the real run.

### What I ran — commands + config; what worked / gotchas
- [ ] `pytest -q tests/` — mechanics (corpus/tools/data/trajectory) pass.
- [ ] `python -c "import data; data.selfcheck()"` — fixture loads, retrieval hits gold.
- [ ] (next increment) `bash launch_cloud.sh sanity` — rLLM+veRL pipeline green on 0.5B.
- [ ] (next increment) RunPod real run + eval gate.
- Gotchas (record the ones that actually bit): ___

### Results — curves, eval numbers, before/after; honest read
- **Eval gate (held-out, base ReAct vs GRPO-tuned):**

  | metric | base | tuned | Δ |
  |---|---|---|---|
  | answer EM | | | |
  | answer F1 | | | |
  | retrieval hit-rate (gold titles found) | | | |
  | groundedness (judge) | | | |
  | avg #steps | | | |
  | judge-grounded gap (hacking probe) | | | |

- Gate result: ___ (PASS only if answer gain **and** groundedness holds — no judge-gaming).
- Honest read (real gain vs verbosity / fake citations / over-searching): ___

### What I learned — 2–4 bullets
- ___

## Run it — exact commands
```bash
python -m venv .venv-deep-research && source .venv-deep-research/bin/activate
pip install -r requirements.txt

pytest -q tests/                     # mechanics: corpus/tools/data/trajectory (no GPU/model)
python -c "import data; data.selfcheck()"   # fixture + retrieval sanity (offline)
# (next increment) rLLM+veRL training + eval gate — see HANDOFF.md / launch_cloud.sh
```

## Current-landscape check (web, 2026-08-10) — field moves monthly, so this is dated
- **Base stack still current:** Search-R1 (`PeterGriffinJin/Search-R1`) + veRL is still the
  canonical offline-corpus deep-research RL lab; HotpotQA/2Wiki still standard multi-hop. rLLM
  now also has a **Tinker** training backend + multi-agent training (docs: rllm-project.readthedocs.io).
- **Groundedness / anti-hacking (shapes reward TODO #1):** **Proof-of-Use** (arXiv 2510.10931) —
  citation requirement + evidence-attribution verification + passage-claim entailment. The
  current answer to "did it actually use what it retrieved?" → aim the reward here (citation
  faithfulness = the standout hook), not just judge+overlap.
- **Advanced rollout (stretch/alternate):** **Tree-GRPO** (ICLR 2026, on Search-R1+veRL) —
  tree-search ReAct rollouts on multi-hop QA. **DEEP-GRPO** reports best HotpotQA/2Wiki.
- **Throughput / curriculum:** **SkyRL-Agent** (2511.16108, multi-turn RL efficiency);
  **LiteResearcher** (4B difficulty-aware GRPO curriculum, 71.3% GAIA-Text) — for Branch-A stretch.
- **Domain-corpus template (SEC-10K stretch):** **PaperSearchQA** (2601.18207) — search+reason
  over a scientific-paper corpus with RLVR; a direct pattern for an original domain corpus.
- **Orientation:** Deep Research Agents survey (2506.18096); Model-Native Agentic AI survey (2510.16720).

## Status
**Full-build complete, ready for RunPod.** Core logic (reward, trajectory/credit-assignment
seam, runnable masking verifier), the veRL trainer entry (`train_dr.py`, with a no-rllm
`--dry-run`), the framework-agnostic eval gate (`evaluate.py`), and cloud scripts are all
written. **36 offline tests pass** (`pytest -q tests/`). What remains is pod-side: resolve the
rLLM/veRL version matrix + wire the `# VERIFY` framework seams, run the sanity spike, pass the
masking gate, then the real Qwen2.5-3B run + eval gate. **The RunPod Claude Code session picks
this up from [`HANDOFF.md`](HANDOFF.md).** Design decisions: [`NOTES.md`](NOTES.md).

### What's mine (Harpreet) to review / retune
Full-build wrote the reward from the design we worked out together (NOTES.md). The policy is
config-knob'd so you can retune without editing code: `reward_mode` (gated|additive), `beta`,
`w_fab`, `reward_kind` (em|f1), `citation_backend`, the `lambda_eff` ramp. If you'd rather
re-derive `reward.reward_deep_research` yourself, it's isolated in `reward.py`.