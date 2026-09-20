"""Batched native-tool-calling rollouts for Qwen2.5-Instruct — the FIXED harness.

Replaces the rollout half of `diagnosis1_native_tools.py`, which measured six real bugs
rather than the model (see RFT_PLAN_AND_MODEL_DIAGNOSIS.md's "Harness diagnosis
2026-08-26" section, and `verify_format.py` for the script that found them). Scoring is
deliberately NOT reimplemented here — it still comes from citations.py / metrics.py, so
"correct" and "well-cited" mean exactly what they mean in training.

WHAT THE MEASUREMENTS SAID, AND WHAT EACH FIX DOES ABOUT IT
(every number below is from a real run — transcripts under rft_diagnosis/transcripts/)

1. 28/36 rounds were plain prose that the old harness rejected. On 2 of 4 probe
   questions the model emitted the EXACT gold answer ('RCD Mallorca', 'yes') and it was
   discarded; once it emitted 'Christopher Reeve [Switching Channels] [Bob Holiday]' —
   correct citation format — also discarded.
   -> FIX: in Qwen's native convention a plain-text turn IS how the model finishes;
      there is no `answer` tool in that convention. We now TREAT PLAIN TEXT AS THE FINAL
      ANSWER. The invented `answer` tool is off by default (`include_answer_tool=False`)
      because advertising it contradicts the convention — but a call to it is still
      honoured if the model makes one, so this never regresses.
      NOTE this does mean a model that answers without ever searching ends its episode
      immediately. That is intentional: it is a real failure we want to SEE (it shows up
      as zero_tool_call_rate), not one to paper over by forcing more rounds.

2. The old rejection message contained the literal string `<tool_call>`. Echoed back in
   a user turn it poisoned the conversation — the model reproduced it with a gear emoji
   (U+2699) substituted for 5 straight rounds, or halted at exactly that point, and
   never recovered.
   -> FIX: `_RETRY_MSG` describes the required format WITHOUT containing that literal
      string. Keep it that way; this is a real, measured failure, not fastidiousness.

3. The model emits MULTIPLE tool calls in one reply (one had search+read+read+
   answer(...)+a fake user turn); the old parser kept the first and silently binned the
   rest — including a complete answer. Qwen's own preamble invites this ("You may call
   one or more functions") and nothing stopped generation at the turn boundary.
   -> FIX: `</tool_call>` is a stop string, so generation ends at the FIRST complete
      call. `include_stop_str_in_output=True` keeps the closing tag in the text, which
      the parser's regex requires — without it the stop string is stripped and every
      call would look truncated. The two settings only work as a pair.

4. Instructions rode in the user message, leaving Qwen's stock "You are Qwen, created by
   Alibaba Cloud" as the real system prompt.
   -> FIX: a proper `role="system"` message.

5. The old code injected `k` into search arguments and fed that back into the history,
   showing the model an argument its own schema did not declare.
   -> FIX: `k` is declared in the schema. The model may set it or omit it.

6. Generation ran one prompt at a time in a Python loop.
   -> FIX: lockstep batching, mirroring `evaluate.run_batched_rollouts` — every still-
      active episode is advanced together in ONE `llm.generate(prompts, ...)` call per
      round. With 16 questions x k=4 that is 64 concurrent episodes per generate call
      instead of 64 sequential ones.

NOT a bug, checked and cleared: turn boundaries. All 36 probe rounds ended naturally
(finish_reason='stop'); none hit the max_new_tokens=256 cap.
ALSO verified correct and deliberately left alone: tool_calls arguments are rendered as
a JSON OBJECT, matching Qwen's own spec line. The OpenAI-canonical JSON-STRING form
renders wrong for this template.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import tools as dr_tools
from trajectory import Trajectory, Step, ToolCall


# --------------------------------------------------------------------------- #
# Tool schemas. `answer` is defined but only ADVERTISED when include_answer_tool
# is on — see the module docstring, fix 1.
# --------------------------------------------------------------------------- #
SEARCH_SCHEMA = {"type": "function", "function": {
    "name": "search",
    "description": "Search the document corpus; returns the top-k passage titles and snippets.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "natural-language search query"},
        # Declared because we USED to inject it silently (fix 5). If the model is going
        # to see `k` in the conversation history, it must be a legal argument.
        "k": {"type": "integer", "description": "how many passages to return (optional)"},
    }, "required": ["query"]}}}

READ_SCHEMA = {"type": "function", "function": {
    "name": "read",
    "description": "Read the full text of one passage by its exact title, as returned by search.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string", "description": "the exact passage title, as returned by search"},
    }, "required": ["title"]}}}

ANSWER_SCHEMA = {"type": "function", "function": {
    "name": "answer",
    "description": ("Commit your final answer. Answer with the SHORTEST possible phrase - "
                    "usually 1-3 words. Cite every passage that supports it by its exact "
                    "title in square brackets. This ends the episode."),
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "description": "the shortest possible answer phrase, with [Title] citations"},
    }, "required": ["text"]}}}


def tool_schemas(include_answer_tool: bool = False) -> list[dict]:
    return [SEARCH_SCHEMA, READ_SCHEMA] + ([ANSWER_SCHEMA] if include_answer_tool else [])


# System prompt (fix 4). The finishing convention is stated explicitly because it is the
# one thing the native format leaves implicit and our scoring depends on.
_BASE_PROMPT = (
    "You are a research agent. Answer the question using the functions provided.\n"
    "\n"
    "How to work:\n"
    "- Call `search` to find relevant passages, then `read` to get a passage's full text.\n"
    "- Read a passage before relying on it. A search snippet alone is not enough.\n"
    "- Call EXACTLY ONE function per turn, then stop and wait for its result. Never write "
    "the result yourself.\n"
)

# fix 8 / the `think` variable. MEASURED 2026-08-26: across every reply that contained a
# tool call, the model wrote NOTHING before it — 0 of 8. Its plain-text answers ran 2-5
# tokens. So in native-calling mode this model does no visible reasoning at all.
#
# That is a real confound in the original bracket-vs-native A/B, not a detail: the
# bracket format is ReAct and has an explicit `Thought:` channel, while Qwen's native
# tool-calling template has no thought channel and nothing asks for one. The two arms
# differed on reasoning as well as on syntax, so the comparison never isolated syntax.
# Made an explicit flag rather than switched on by default, so it can be A/B'd cleanly —
# and note it interacts with the finishing rule below (a bare thought with no function
# call could be misread as a final answer; the transcripts are the check for that).
_THINK_CLAUSE = (
    "- Before each function call, write ONE short sentence saying what you are looking "
    "for and why. Then make the call.\n"
)

_FINISH_CLAUSE = (
    "\n"
    "How to finish:\n"
    "- When you know the answer, reply with plain text and no function call.\n"
    "- That final reply must be the SHORTEST possible phrase - usually 1-3 words: the exact "
    "entity, name, number, or 'yes'/'no'. No sentence, no explanation.\n"
    "- After the phrase, cite every passage that supports it by its exact title in square "
    "brackets, e.g. American [Blue Harvest (film)] [Jane Doe (director)]\n"
    "- Never cite a passage you did not read."
)


def system_prompt(think: bool = False) -> str:
    return _BASE_PROMPT + (_THINK_CLAUSE if think else "") + _FINISH_CLAUSE


SYSTEM_PROMPT = system_prompt(think=False)      # back-compat for existing importers

# Fix 2: describes the format WITHOUT containing the literal opening tag, which is what
# poisoned the conversation when echoed back.
_RETRY_MSG = ("Your last reply could not be read. Either call exactly one function using "
              "the required XML-tagged JSON format, or, if you already know the answer, "
              "reply with plain text only.")

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Fix 3: `</tool_call>` ends the turn at the first complete call. Pair with
# include_stop_str_in_output=True (see build_sampling_params) or the regex never matches.
STOP_STRINGS = ["</tool_call>", "<tool_response>", "<|im_start|>"]


def build_sampling_params(cfg, temperature: float | None = None):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=cfg.temperature if temperature is None else temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        stop=STOP_STRINGS,
        include_stop_str_in_output=True,   # keeps `</tool_call>`; the parser needs it
    )


def parse_reply(text: str) -> tuple[str, dict, bool]:
    """Extract the first complete tool call. Returns (name, args, found).

    `found=False` means no tool call is present — under the native convention that is
    the model FINISHING, not an error (fix 1). The caller decides, not this function.
    """
    m = _TOOL_CALL_RE.search(text or "")
    if not m:
        return "", {}, False
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return "", {}, False
    name = str(obj.get("name", "")).strip().lower()
    args = obj.get("arguments", {})
    if isinstance(args, str):              # some models emit arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return name, {}, False
    if not isinstance(args, dict):
        return name, {}, False
    return name, args, True


@dataclass
class NativeEpisode:
    """One rollout's mutable state. One task may have several (k samples)."""
    task: object
    sample_idx: int
    history: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    done: bool = False
    finish_mode: str = ""            # plain_text | answer_tool | round_limit | dead
    n_retries: int = 0               # unparseable replies, i.e. genuine format failures
    n_thoughts: int = 0              # replies that wrote reasoning before the call (fix 8)
    _docstore: object = None

    def start(self, think: bool = False, examples: bool = False) -> None:
        self._docstore = self.task.docstore()
        self.history = [{"role": "system", "content": system_prompt(think)}]
        if examples:
            # Few-shot as real prior turns, the way a chat model consumes them. Fills the
            # missing 2x2 cell: the bracket arm always carried these, the native arm never
            # did, so bracket's correctness lead could never be attributed to syntax.
            from native_examples import worked_examples
            self.history.extend(worked_examples())
        self.history.append({"role": "user", "content": f"Question: {self.task.question}"})

    def to_trajectory(self, cfg) -> Trajectory:
        return Trajectory(
            task_id=self.task.task_id, query=self.task.question, tools=list(cfg.tools),
            gold_answer=self.task.gold_answer, gold_aliases=list(self.task.gold_aliases),
            supporting_titles=list(self.task.supporting_titles), steps=self.steps,
            final_answer=self.final_answer, done=self.done)


def _advance(ep: NativeEpisode, raw: str, cfg, note=None) -> None:
    """Apply one model reply to one episode. Mirrors the old run_one_episode's bookkeeping
    so trajectory/scoring stay identical — only the DECISIONS changed.

    `note(ep, text)` is an optional hook that records, for the transcript, what the
    harness DID with this reply. It is called after the decision, never before, so it
    reports the real outcome instead of predicting it.
    """
    def _note(msg: str) -> None:
        if note is not None:
            note(ep, msg)

    name, args, found = parse_reply(raw)

    # --- fix 1: no tool call => the model is finishing, in plain text. ---
    if not found:
        stripped = (raw or "").strip()
        if stripped:
            ep.final_answer = stripped
            ep.steps.append(Step(thought="", call=ToolCall("answer", {"text": stripped}, raw=raw),
                                 observation=stripped, ok=True, parse_ok=True))
            ep.done = True
            ep.finish_mode = "plain_text"
            _note("No function call in the reply. Under the native convention that means "
                  "the model is FINISHED, so this text is taken as its final answer and "
                  "the episode ends here.")
            return
        # Genuinely empty reply — nothing to interpret. Ask once more.
        ep.n_retries += 1
        ep.steps.append(Step(thought="", call=None, observation="error: empty reply",
                             ok=False, error="empty", parse_ok=False))
        ep.history.append({"role": "assistant", "content": raw})
        ep.history.append({"role": "user", "content": _RETRY_MSG})
        _note("Reply was empty. Asked the model to try again (costs one round).")
        return

    if name not in dr_tools.TOOLS:
        # A tool call we can read but cannot honour: malformed JSON, or an invented name.
        ep.n_retries += 1
        ep.steps.append(Step(thought="", call=None,
                             observation=f"error: unknown or malformed call {name!r}",
                             ok=False, error="parse", parse_ok=False))
        ep.history.append({"role": "assistant", "content": raw})
        ep.history.append({"role": "user", "content": _RETRY_MSG})
        _note(f"Called {name!r}, which is not one of our functions (or the JSON was "
              "malformed). Asked the model to try again (costs one round).")
        return

    # fix 7: keep any reasoning written BEFORE the call. The template renders assistant
    # `content` alongside `tool_calls` ({%- if message.content %}), so dropping it — as
    # the old code did by omitting the field — deleted the model's own thinking from the
    # history, leaving it unable to build on it across turns.
    thought = raw[: raw.find("<tool_call>")].strip() if "<tool_call>" in raw else ""
    if thought:
        ep.n_thoughts += 1

    if name == "answer":
        # Honoured even when not advertised — never regress on a model that uses it.
        text = str(args.get("text", "")).strip()
        ep.final_answer = text
        ep.steps.append(Step(thought=thought, call=ToolCall(name, args, raw=raw),
                             observation=text, ok=True, parse_ok=True))
        ep.done = True
        ep.finish_mode = "answer_tool"
        _note(f"Model called the `answer` function. Episode ends. text={text!r}")
        return

    if name == "search" and "k" not in args:
        args = {**args, "k": cfg.search_k}          # legal now — declared in the schema
    obs, ok, err, titles = dr_tools.execute(ep._docstore, name, args)
    obs = obs[: cfg.max_obs_chars]
    ep.steps.append(Step(thought=thought, call=ToolCall(name, args, raw=raw), observation=obs,
                         ok=ok, error=err, parse_ok=True, retrieved_titles=titles))
    assistant_msg: dict = {"role": "assistant",
                           "tool_calls": [{"function": {"name": name, "arguments": args}}]}
    if thought:
        assistant_msg["content"] = thought
    ep.history.append(assistant_msg)
    ep.history.append({"role": "tool", "content": obs})
    _note(f"Ran {name}({args}) -> ok={ok}"
          + (f", err={err}" if err else "")
          + (f", titles={titles}" if titles else "")
          + "\nSent the result back to the model as a tool response (shown below).\n"
          + "TOOL RESULT:\n" + obs)


def run_batched_native_rollouts(tasks, cfg, llm, tokenizer, sampling_params, k: int = 1,
                                include_answer_tool: bool = False,
                                max_rounds: int | None = None,
                                think: bool = False, examples: bool = False,
                                on_round=None, on_action=None) -> list[NativeEpisode]:
    """Roll `k` episodes per task, ALL CONCURRENTLY (fix 6).

    One `llm.generate()` call per round over every still-active episode — the same
    lockstep shape as evaluate.run_batched_rollouts, which is the pattern this lab
    already proved on a pod. Never call this per-episode in a loop.

    `on_round(round_idx, episodes, raws)` is an optional hook for transcript capture.
    """
    schemas = tool_schemas(include_answer_tool)
    max_rounds = max_rounds if max_rounds is not None else int(getattr(cfg, "max_turns", 8)) + 1

    episodes = [NativeEpisode(task=t, sample_idx=i) for t in tasks for i in range(k)]
    for ep in episodes:
        ep.start(think, examples)

    for rnd in range(max_rounds):
        active = [ep for ep in episodes if not ep.done]
        if not active:
            break
        prompts = [tokenizer.apply_chat_template(
            ep.history, tools=schemas, add_generation_prompt=True, tokenize=False)
            for ep in active]
        outs = llm.generate(prompts, sampling_params, use_tqdm=False)
        raws = [o.outputs[0].text for o in outs]
        if on_round is not None:
            on_round(rnd, active, prompts, raws, outs)
        for ep, raw in zip(active, raws):
            _advance(ep, raw, cfg, note=on_action)

    for ep in episodes:
        if not ep.done:
            # Ran out of rounds without ever finishing. Distinguish "explored but never
            # concluded" from "never engaged" — different diagnoses, per the plan doc.
            ep.finish_mode = "round_limit" if ep.steps else "dead"
    return episodes
