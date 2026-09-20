"""DeepResearchEnv — the rLLM Environment that wraps the corpus + tools + reward.

This is the ADAPTER between rLLM/veRL and the deep-research learning logic. It
reuses (does NOT reimplement):
    corpus.py : DocStore, BM25 retriever   — the information environment
    tools.py  : execute(), TOOLS           — the search/read/answer sandbox
    data.py   : load_tasks(), DRTask       — the multi-hop QA data
    reward.py : reward_deep_research()      — the layered reward (YOURS, TODO #1)
    judge.py  : make_judge()               — the groundedness judge (frozen)

The gym contract rLLM expects (VERL_RLLM_PRIMER Part-2 Page 5):
    reset()      -> (observation, info)
    step(action) -> (observation, reward, done, info)
    from_dict(env_args) -> DeepResearchEnv        # rLLM builds a fresh env per rollout

One question = one episode: the agent searches the DocStore for that question,
reads passages, and commits an answer; the terminal reward scores it.

⚠️  Every rllm import below is `# VERIFY` — the base-class module path
    (rllm.environments.base.base_env) and the exact step/reset return contract have
    changed across rllm versions. Confirm against your installed version (HANDOFF.md
    step 1) before trusting this file. Same discipline as finqa_agent/verl/env.py.
"""
from __future__ import annotations

import re

import tools as dr_tools           # execute, TOOLS
import reward as dr_reward         # reward_deep_research (yours)
from trajectory import Trajectory, Step, ToolCall

# from rllm.environments.base.base_env import BaseEnv     # VERIFY module path
BaseEnv = object   # placeholder so the file imports without rllm installed; SWAP on pod.


class DeepResearchEnv(BaseEnv):
    """One multi-hop question = one episode over a fixed DocStore."""

    def __init__(self, task, docstore, reward_cfg, judge=None, max_turns: int = 8,
                 search_k: int = 3, max_obs_chars: int = 1200, global_step: int = 0):
        self.task = task                                # a DRTask
        self.world = docstore                           # DocStore for THIS question
        self.reward_cfg = reward_cfg                    # the Config (reward knobs + ramp)
        self.judge = judge
        self.max_turns = max_turns
        self.search_k = search_k
        self.max_obs_chars = max_obs_chars
        self._global_step = global_step                 # for the RAMPED efficiency toll
        self._turns = 0
        self._steps: list[Step] = []                    # accumulates the ReAct turns
        self._history: list[dict] = []                  # the raw rLLM conversation
        self._final_answer: str | None = None
        self._done = False

    def set_global_step(self, step: int) -> None:
        """Trainer calls this so lambda_eff can ramp with training progress. See the
        `# VERIFY` note in train_dr.py on how verl exposes the current step to the env."""
        self._global_step = int(step)

    # -- rLLM constructs envs from a dict, once per rollout (primer Page 5) --
    @staticmethod
    def from_dict(env_args: dict) -> "DeepResearchEnv":
        task = env_args["task"]
        cfg = env_args["cfg"]
        return DeepResearchEnv(
            task=task,
            docstore=task.docstore(),                   # bundled per-question corpus
            reward_cfg=cfg,
            judge=env_args.get("judge"),
            max_turns=getattr(cfg, "max_turns", 8),
            search_k=getattr(cfg, "search_k", 3),
            max_obs_chars=getattr(cfg, "max_obs_chars", 1200),
            global_step=env_args.get("global_step", 0),
        )

    def reset(self):
        self._turns = 0
        self._steps = []
        self._history = []
        self._final_answer = None
        self._done = False
        obs = _opening_prompt(self.task, self.reward_cfg)
        self._history.append({"role": "user", "content": obs})
        return obs, {}

    def step(self, action):
        """action = the model's assistant turn (raw text). Plumbing is scaffolded;
        the terminal reward + trajectory conversion are TODO(harpreet)."""
        self._turns += 1
        text = _action_text(action)
        self._history.append({"role": "assistant", "content": text})

        thought, name, args, parse_ok = _parse_react_action(text)

        # ---- terminal: model committed an answer, OR we hit the turn cap ----
        if name == "answer" and parse_ok:
            self._final_answer = str(args.get("text", "")).strip()
            self._steps.append(Step(thought=thought, call=ToolCall(name, args, raw=text),
                                    observation=self._final_answer, ok=True, parse_ok=True))
            self._done = True
            reward, info = self._terminal_reward()
            return "", reward, True, info

        if self._turns >= self.max_turns:
            # forced stop without an answer — still score it (reward penalizes no-answer)
            if parse_ok and name in dr_tools.TOOLS:
                self._run_tool(thought, name, args, text)   # record the last action too
            reward, info = self._terminal_reward()
            return "", reward, True, info

        # ---- non-terminal: run the tool, feed the observation back ----
        obs = self._run_tool(thought, name, args, text)
        self._history.append({"role": "user", "content": obs})
        return obs, 0.0, False, {"tool": name, "turn": self._turns}

    # ------------------------------------------------------------------ #
    #  tool execution + observation shaping (SCAFFOLD — mechanics)         #
    # ------------------------------------------------------------------ #
    def _run_tool(self, thought, name, args, raw) -> str:
        if name not in dr_tools.TOOLS:
            obs = f"error: unknown or unparsed action. Use one of: {', '.join(dr_tools.TOOLS)}."
            self._steps.append(Step(thought=thought, call=None, observation=obs,
                                    ok=False, error="parse", parse_ok=False))
            return obs
        if name == "search" and "k" not in args:
            args = {**args, "k": self.search_k}
        obs, ok, err, titles = dr_tools.execute(self.world, name, args)
        obs = obs[: self.max_obs_chars]                 # bound the tool-token span (masking!)
        self._steps.append(Step(thought=thought, call=ToolCall(name, args, raw=raw),
                                observation=obs, ok=ok, error=err, parse_ok=True,
                                retrieved_titles=titles))
        return obs

    # ------------------------------------------------------------------ #
    #  the credit-assignment seam  (full-build 2026-08-14)                 #
    # ------------------------------------------------------------------ #
    def _to_dr_trajectory(self) -> Trajectory:
        """Assemble the `Trajectory` reward_deep_research scores, from the per-turn
        Steps recorded during the episode. This decides WHAT gets graded.

        NOTE on tokens: input_ids / model_mask are filled by veRL's multi-turn
        tokenization (the trainer), NOT here — this trajectory is the *semantic* record
        the reward reasons over (answer, citations, retrieved titles, steps). The
        token-level mask that gates the loss is a separate artifact; assert_verl_masking
        _matches() is the check that veRL marked only model tokens (never the retrieved
        passages) — the L4 silent bug."""
        return Trajectory(
            task_id=self.task.task_id,
            query=self.task.question,
            tools=list(dr_tools.TOOLS),
            gold_answer=self.task.gold_answer,
            gold_aliases=list(self.task.gold_aliases),
            supporting_titles=list(self.task.supporting_titles),
            steps=self._steps,
            final_answer=self._final_answer,
            done=self._done,
            meta={"n_turns": self._turns, "global_step": self._global_step,
                  "corpus_size": len(getattr(self.world, "docs", {}) or {})},
        )

    def _terminal_reward(self):
        """Score the finished episode. Returns (reward_float, info_dict) — the info dict
        is the per-trajectory breakdown train/eval log so the hacking probes can fire.

        The RAMPED lambda_eff needs the *global* training step: we read self._global_step,
        set via from_dict's env_args['global_step']. RESOLVED 2026-08-24 (was a real, silent
        gap — always 0, toll permanently off regardless of step count): rllm_workflow.py's
        DeepResearchWorkflow.reset() now sources this from a module-level counter kept live
        by a patched AgentWorkflowEngine.set_training_step — see that patch's docstring in
        rllm_workflow.py for the full mechanism. Still defaults to 0 here if from_dict is
        ever called without a 'global_step' key (e.g. a future direct-construction path) —
        toll-off is the deliberately safe fallback, not a design assumption anymore."""
        traj = self._to_dr_trajectory()
        info = dr_reward.reward_deep_research(
            self.task, traj, self.judge, self.reward_cfg, step=self._global_step)
        info_dict = {
            "reward": info.reward, "correct": info.correct, "outcome": info.outcome,
            "groundedness": info.groundedness, "cite_f1": info.cite_f1,
            "cite_precision": info.cite_precision, "cite_recall": info.cite_recall,
            "cite_fabricated": info.cite_fabricated, "hit_rate": info.hit_rate,
            "judge": info.judge, "format_error": info.format_error,
            "n_steps": info.n_steps, **info.components,
            # 2026-09-07: behaviour counters for the RL-from-SFT run (mechanics, not
            # reward). `capped` = the episode ran out its turn budget — the never-commit
            # failure RL is targeting; `answered` = a parsed answer[...] was emitted.
            "n_reads": sum(1 for c in traj.tool_calls if c.name == "read"),
            "n_searches": traj.n_searches,
            "n_tool_calls": len(traj.tool_calls),
            "answered": 1.0 if traj.done else 0.0,
            "capped": 0.0 if traj.done else 1.0,
        }
        return info.reward, info_dict


# ---------------------------------------------------------------------------- #
#  helpers (SCAFFOLD — mechanics, safe to rely on)                             #
# ---------------------------------------------------------------------------- #
def _action_text(action) -> str:
    """rLLM may pass an Action object or a raw string — normalize. VERIFY the type."""
    return getattr(action, "action", action) if action is not None else ""


# ReAct action grammar the agent emits. Kept deliberately forgiving (the reward's
# format_error prices sloppiness; the parser shouldn't hard-fail a recoverable turn):
#   Thought: <free text>
#   Action: <tool>[<arg-or-json>]        e.g.  search[who directed Blue Harvest]
#                                              read[Blue Harvest (film)]
#                                              answer[American]
#
# 2026-08-25, real bug found running rft_diagnosis/diagnosis1.py against the actual
# Qwen2.5-3B-Instruct model (not caught by any prior sanity check — the tiny 0.5B
# sanity stand-in rarely over-generates far enough to trigger it): with NOTHING
# stopping generation at the true turn boundary, the model sometimes free-runs past
# its first `Action:` and writes an entire HALLUCINATED multi-turn continuation in
# one completion (fake "search results:", fake `read` calls, a final `answer`) —
# imitating the worked examples' full-trajectory SHAPE rather than stopping after
# one step. The OLD regex (`re.DOTALL`, greedy `(.*)`, anchored to `$`) doesn't just
# fail on this — it silently ACCEPTS it: greedy `.*` under DOTALL matches across
# every line, so `re.search` locks onto the FIRST "Action:" but captures ALL THE WAY
# to the LAST `]` in the entire remaining text, stuffing every hallucinated
# subsequent line into that turn's argument string (verified directly: a
# `search[...]` call ended up with the ENTIRE fake rest-of-trajectory, including a
# fake final answer, crammed into its `query` string — silently corrupting the
# real tool call rather than raising a visible error). When the hallucinated blob
# instead gets cut off mid-way by `max_new_tokens`, the same over-eager `$`-anchor
# fails to match AT ALL (no trailing `]` at the truncated string's end) — the OTHER
# failure mode, a spurious parse failure for a turn that actually started fine.
# Fix: confine the match to a SINGLE LINE (`re.MULTILINE`, no `DOTALL`, so `.`
# no longer crosses newlines) — `re.search` then finds only the FIRST complete
# `Action: tool[...]` line and ignores everything hallucinated after it, which is
# the actually-intended "one action per turn" semantics. Still handles legitimate
# nested brackets on ONE line correctly (e.g. `answer[American [Blue Harvest
# (film)] [Jane Doe (director)]]` — greedy `.*` still grabs to the LAST `]` on
# that same line, just never crosses into a different line/turn.
_ACTION_RE = re.compile(r"Action:\s*([a-zA-Z_]+)\s*\[(.*)\]\s*$", re.MULTILINE)
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?:\nAction:|\Z)", re.DOTALL)


def _parse_react_action(text: str):
    """Parse one assistant turn -> (thought, tool_name, args_dict, parse_ok).

    Mechanics only (no learning logic). Maps the single bracket payload to the
    tool's primary arg: search->query, read->title, answer->text. `k` for search
    is injected by the env from config. Returns parse_ok=False (name="") when no
    legal Action line is found, so the env can hand the agent a recoverable error."""
    thought_m = _THOUGHT_RE.search(text or "")
    thought = thought_m.group(1).strip() if thought_m else ""
    m = _ACTION_RE.search(text or "")
    if not m:
        return thought, "", {}, False
    name = m.group(1).strip().lower()
    payload = m.group(2).strip()
    primary = {"search": "query", "read": "title", "answer": "text"}.get(name)
    if primary is None:
        return thought, name, {}, False        # unknown tool — env returns an error obs
    return thought, name, {primary: payload}, True


def _opening_prompt(task, cfg) -> str:
    """The ReAct system/opening prompt: the question + the tool menu + the format.
    Mechanics — tune the wording freely; it's not a learning artifact."""
    from tools import render_toolset
    cite = getattr(cfg, "require_citations", True)
    cite_rule = (
        "Ground your answer ONLY in passages you retrieved. CITE EVERY passage that "
        "supports your answer — usually ONE PER FACT/HOP — by its exact title in square "
        "brackets. Multi-hop answers need MULTIPLE citations, e.g. "
        "answer[American [Blue Harvest (film)] [Jane Doe (director)]]. Do not cite a "
        "passage you did not read.\n"
        if cite else "")
    cite_example = ("Action: answer[American [Blue Harvest (film)] [Jane Doe (director)]]"
                    if cite else "Action: answer[American]")
    # 2026-08-24: added a FULL worked multi-hop example (search->read->search->read->
    # answer), not just isolated search/answer syntax snippets. Root cause found this
    # session (TRAINING_HISTORY_LOG.md "eval-time turn count" entry): every healthy run
    # converged on ~exactly 2 turns (one search, then answer directly) regardless of
    # reward design — the prompt told the model in prose to "search, READ, then answer"
    # but never demonstrated a `read` call or the actual multi-hop loop, only isolated
    # search/answer syntax. Reuses the SAME fictional Blue Harvest/Jane Doe entities
    # already in the existing citation example (no new content to learn, just the
    # missing demonstrated PATTERN) and explicitly ties each citation back to a read()
    # call, reinforcing "Do not cite a passage you did not read" with a worked case
    # instead of only stating the rule. Kept concise — token budget is tight
    # (max_new_tokens=256/turn) and this is a worked EXAMPLE for a DIFFERENT question,
    # not part of the live conversation.
    #
    # 2026-08-25: found a SECOND, deeper gap in the same example (Harpreet's catch,
    # while reviewing rft_diagnosis/diagnosis1.py's worked examples) — every
    # search/read Action was followed IMMEDIATELY by another Thought line, never by
    # the actual tool-response text a real rollout would insert there (the
    # "search results:\n[1] Title: snippet..." / "[Title]\n<full text>" turn
    # tools.py really produces, injected as its own role=user message by
    # env.step()). The 2026-08-24 fix taught the SHAPE of the loop (search, read,
    # search, read, answer) but the example itself never showed what the model is
    # actually reacting to at each step — it narrated conclusions ("A passage
    # titled X names Y") as if already known, rather than showing the retrieved
    # text and reasoning from it. Fixed by inserting the real tool-response text
    # (verbatim in tools.py's exact format) after every Action, so the example now
    # matches the true shape of a live rollout, turn for turn.
    # Stripped for SFT and every stage after it — see cfg.include_worked_example.
    if not getattr(cfg, "include_worked_example", True):
        return (
            "You are a research agent. Answer the question by SEARCHING a document "
            "corpus, READING the most relevant passages, then giving a short, grounded "
            "final answer. Base your answer only on what you retrieve.\n"
            "Answer with the SHORTEST possible phrase — usually 1-3 words: the exact "
            "entity, name, number, or 'yes'/'no'. No sentence, no explanation.\n"
            + cite_rule + "\n"
            "Tools:\n" + render_toolset(list(cfg.tools)) + "\n\n"
            "Format each turn as:\nThought: <your reasoning>\nAction: <tool>[<argument>]\n"
            "Write EXACTLY ONE Thought/Action pair, then STOP — do not write what the "
            "tool returns yourself; the real result will be given to you before your "
            "next turn.\n"
            f"\nQuestion: {task.question}"
        )

    worked_example = (
        "\nWorked example for a different 2-hop question "
        "(\"What nationality is the director of the film Blue Harvest?\"):\n"
        "Thought: First I need to find who directed Blue Harvest.\n"
        "Action: search[who directed the film Blue Harvest]\n"
        "search results:\n"
        "[1] Blue Harvest (film): Blue Harvest is a 2009 drama film. It was directed "
        "by Jane Doe and premiered at the Toronto festival. The film follows a "
        "family farm over one summer.\n"
        "[2] Blue Harvest (album): Blue Harvest is a 2015 studio album by the "
        "Norwegian band Fjord. It reached number three on the domestic chart.\n"
        "[3] John Roe (producer): John Roe is a British film producer who has "
        "worked on several independent dramas.\n"
        "Thought: [1] is the film, and it names Jane Doe as director. Reading it "
        "to confirm before citing.\n"
        "Action: read[Blue Harvest (film)]\n"
        "[Blue Harvest (film)]\n"
        "Blue Harvest is a 2009 drama film. It was directed by Jane Doe and "
        "premiered at the Toronto festival. The film follows a family farm over "
        "one summer.\n"
        "Thought: Confirmed — Jane Doe directed it. Now I need her nationality.\n"
        "Action: search[Jane Doe director nationality]\n"
        "search results:\n"
        "[1] Jane Doe (director): Jane Doe is an American film director, born in "
        "Ohio in 1971. She is known for independent dramas including Blue Harvest "
        "and Winter Wheat.\n"
        "[2] Blue Harvest (film): Blue Harvest is a 2009 drama film. It was "
        "directed by Jane Doe and premiered at the Toronto festival.\n"
        "[3] John Roe (producer): John Roe is a British film producer who has "
        "worked on several independent dramas.\n"
        "Thought: [1] says she is American. Reading it to confirm before citing.\n"
        "Action: read[Jane Doe (director)]\n"
        "[Jane Doe (director)]\n"
        "Jane Doe is an American film director, born in Ohio in 1971. She is "
        "known for independent dramas including Blue Harvest and Winter Wheat.\n"
        "Thought: Confirmed. I have both facts, each grounded in a passage I read.\n"
        "Action: answer[American [Blue Harvest (film)] [Jane Doe (director)]]\n"
        if cite else
        "\nWorked example for a different 2-hop question "
        "(\"What nationality is the director of the film Blue Harvest?\"):\n"
        "Thought: First I need to find who directed Blue Harvest.\n"
        "Action: search[who directed the film Blue Harvest]\n"
        "search results:\n"
        "[1] Blue Harvest (film): Blue Harvest is a 2009 drama film. It was directed "
        "by Jane Doe and premiered at the Toronto festival. The film follows a "
        "family farm over one summer.\n"
        "[2] Blue Harvest (album): Blue Harvest is a 2015 studio album by the "
        "Norwegian band Fjord. It reached number three on the domestic chart.\n"
        "[3] John Roe (producer): John Roe is a British film producer who has "
        "worked on several independent dramas.\n"
        "Thought: [1] is the film, and it names Jane Doe as director. Reading it "
        "to confirm.\n"
        "Action: read[Blue Harvest (film)]\n"
        "[Blue Harvest (film)]\n"
        "Blue Harvest is a 2009 drama film. It was directed by Jane Doe and "
        "premiered at the Toronto festival. The film follows a family farm over "
        "one summer.\n"
        "Thought: Confirmed — Jane Doe directed it. Now I need her nationality.\n"
        "Action: search[Jane Doe director nationality]\n"
        "search results:\n"
        "[1] Jane Doe (director): Jane Doe is an American film director, born in "
        "Ohio in 1971. She is known for independent dramas including Blue Harvest "
        "and Winter Wheat.\n"
        "[2] Blue Harvest (film): Blue Harvest is a 2009 drama film. It was "
        "directed by Jane Doe and premiered at the Toronto festival.\n"
        "[3] John Roe (producer): John Roe is a British film producer who has "
        "worked on several independent dramas.\n"
        "Thought: [1] says she is American. Reading it to confirm.\n"
        "Action: read[Jane Doe (director)]\n"
        "[Jane Doe (director)]\n"
        "Jane Doe is an American film director, born in Ohio in 1971. She is "
        "known for independent dramas including Blue Harvest and Winter Wheat.\n"
        "Thought: Confirmed.\nAction: answer[American]\n"
    )
    return (
        "You are a research agent. Answer the question by SEARCHING a document "
        "corpus, READING the most relevant passages, then giving a short, grounded "
        "final answer. Base your answer only on what you retrieve.\n"
        "Answer with the SHORTEST possible phrase — usually 1-3 words: the exact "
        "entity, name, number, or 'yes'/'no'. No sentence, no explanation.\n"
        + cite_rule + "\n"
        "Tools:\n" + render_toolset(list(cfg.tools)) + "\n\n"
        "Format each turn as:\nThought: <your reasoning>\nAction: <tool>[<argument>]\n"
        "Write EXACTLY ONE Thought/Action pair, then STOP — do not write what the "
        "tool returns yourself; the real result will be given to you before your "
        "next turn.\n"
        + worked_example +
        f"\nQuestion: {task.question}"
    )


# ---------------------------------------------------------------------------- #
#  MASKING CHECK — the one silent-failure risk of the whole build              #
#  (full-build 2026-08-14 — a REAL runnable check, not a stub)                  #
# ---------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def decode_where(tokenizer, input_ids, mask, value: int) -> str:
    """Decode the CONTIGUOUS runs of `input_ids` where `mask == value`, joined by
    ' | '. veRL builds the loss mask per assistant span, so runs are contiguous and
    decode cleanly. Used to eyeball what the trainer will (mask==1) and won't (mask==0)
    grade. `tokenizer` needs a `.decode(list[int]) -> str` (HF tokenizers qualify)."""
    runs, cur = [], []
    for tid, m in zip(list(input_ids), list(mask)):
        if int(m) == value:
            cur.append(int(tid))
        elif cur:
            runs.append(tokenizer.decode(cur)); cur = []
    if cur:
        runs.append(tokenizer.decode(cur))
    return " | ".join(runs)


def assert_verl_masking_matches(tokenizer, input_ids, loss_mask, trajectory,
                                min_probe_chars: int = 24, verbose: bool = True) -> dict:
    """Verify veRL's multi-turn loss mask grades ONLY model-generated tokens — never
    the retrieved passages or the opening prompt. THE silent-failure guard of the whole
    build: a wrong mask here trains the model on copied Wikipedia and the loss curve
    looks perfectly healthy while it learns garbage.

    Pull ONE real rollout on the pod (train_dr.py has a `--mask-check` hook), hand its
    veRL-produced (input_ids, loss_mask) plus the DeepResearchEnv trajectory, and this:
      1. decodes the mask==1 (graded) text and the mask==0 (ignored) text,
      2. asserts each RETRIEVED passage's distinctive prefix appears in the IGNORED
         text and NOT in the graded text (the core check),
      3. asserts at least some assistant answer/thought text IS in the graded text
         (catches the opposite bug: everything masked out → no learning signal).

    Raises AssertionError with a readable diff on failure; returns a small report on
    success. Cross-checks the finqa discipline (../finqa_agent/masking.py)."""
    graded = _norm(decode_where(tokenizer, input_ids, loss_mask, 1))
    ignored = _norm(decode_where(tokenizer, input_ids, loss_mask, 0))

    # (2) retrieved-passage tokens must be IGNORED, never graded
    leaked = []
    checked = 0
    for s in trajectory.steps:
        if s.call is None or s.call.name not in ("search", "read") or not s.ok:
            continue
        body = s.observation
        if body.startswith("[") and "]\n" in body:      # strip the "[title]\n" prefix
            body = body.split("]\n", 1)[1]
        probe = _norm(body)[:min_probe_chars]
        if len(probe) < min_probe_chars:
            continue
        checked += 1
        if probe in graded:
            leaked.append(probe)
    assert not leaked, (
        "MASK BUG: retrieved-passage text is being GRADED (mask==1). The model would be "
        "trained on tokens it copied, not generated. Leaked probes:\n  - "
        + "\n  - ".join(leaked[:5])
        + "\nFix veRL's multi-turn mask so tool-return spans are 0. See HANDOFF.md.")

    # (3) at least some model-written answer text must be GRADED (catches mask-all-0).
    # Use the citation-stripped answer's first word — the raw answer carries [Title]
    # markers that won't tokenize/normalize identically to the graded span.
    import citations
    ans = _norm(citations.strip_citations(trajectory.final_answer or ""))
    first_word = ans.split()[0] if ans else ""
    model_signal_present = bool(graded) and (first_word in graded if first_word else True)
    assert model_signal_present, (
        "MASK BUG: no model-generated text is graded (mask all 0?) — there is no learning "
        "signal. Expected the assistant's answer/thought spans to have mask==1.")

    report = {"checked_passages": checked, "leaked": len(leaked),
              "graded_chars": len(graded), "ignored_chars": len(ignored)}
    if verbose:
        print(f"[mask-check] OK — {checked} retrieved passages, none graded; "
              f"model text present in graded span. {report}")
    return report
