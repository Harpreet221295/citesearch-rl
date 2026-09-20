"""Multi-turn trajectory -> masked SFT example. The part that has to be exactly right.

Harpreet's requirement, verbatim (2026-08-26): *"we also have to make sure masks n all
are collected well for SFT, we don't want to train model on tool responses or our
prompts, only on model's own generations."*

That is the whole job of this file. A trajectory is a conversation:

    user      <- our opening prompt (question + tool menu + format)   MASKED
    assistant <- "Thought: ...\\nAction: search[...]"                  GRADED
    user      <- "search results:\\n[1] ..."   (OUR tool output)       MASKED
    assistant <- "Thought: ...\\nAction: read[...]"                    GRADED
    user      <- "[Title]\\n<passage text>"    (OUR tool output)       MASKED
    assistant <- "Thought: ...\\nAction: answer[... [Cite]]"           GRADED

Only the assistant spans carry loss. Everything else is `-100`.

WHY IT MATTERS MORE THAN IT LOOKS. Training on tool output teaches the model to
hallucinate search results — it learns to *produce* passages instead of *retrieving*
them, which is the precise opposite of a research agent, and it would look fine in the
loss curve while doing it. Masking bugs are the silent kind: nothing crashes, the number
goes down, the model is wrong. This lab already has that lesson written down twice
(`env.py::assert_verl_masking_matches`, `WORKFLOW_PORT_NOTES.md`'s masking discovery).

HOW THE BOUNDARY IS FOUND — prefix-delta, the same principle `assignments/sft_min/data.py`
uses, extended from one turn to many:

    prefix_ids = apply_chat_template(history[:i], add_generation_prompt=True)
    full_ids   = apply_chat_template(history[:i+1], add_generation_prompt=False)
    graded span = full_ids[len(prefix_ids):]

The assistant's tokens are exactly what the template adds when that turn is appended.
This is derived from the tokenizer rather than assumed, so it cannot drift if the chat
template changes — which is the trap a hand-written "find the assistant marker" scheme
falls into.

`sft_min`'s hard-won caveat is carried over: **assert the prompt render is a genuine
prefix of the full render.** Some templates are not prefix-stable (they re-render
earlier turns when a later one is appended). We compute the real common prefix, compare
it to `len(prompt_ids)`, and COUNT the mismatches rather than trusting it silently — a
non-zero count is a red flag to investigate, not a warning to scroll past.

The end-of-turn token IS graded, deliberately. It is the stop signal; a model that never
learns to emit it never stops generating. `sft_min`'s README makes the same point.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

IGNORE_INDEX = -100          # torch cross-entropy's default ignore value


def _ids(out) -> list[int]:
    """apply_chat_template's return type varies by transformers version: older returns
    list[int], 5.x returns a BatchEncoding. Normalized here, same as sft_min.data."""
    if hasattr(out, "keys") and "input_ids" in out:
        return list(out["input_ids"])
    return list(out)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]
    task_id: str = ""
    n_graded_spans: int = 0            # assistant turns actually carrying loss
    prefix_mismatches: int = 0         # template not prefix-stable at this many turns
    n_rejected_skipped: int = 0        # assistant turns the env rejected -> not graded

    @property
    def n_graded_tokens(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)

    @property
    def graded_fraction(self) -> float:
        return self.n_graded_tokens / max(1, len(self.input_ids))


def is_rejected_turn(history: list[dict], i: int) -> bool:
    """Did the environment reject assistant turn `i` as unparseable?

    Detected from the NEXT turn: env._run_tool answers an unparsed action with
    "error: unknown or unparsed action...". Measured in the first collected batch: the
    teacher sometimes writes `Answer: X [Cite]` instead of `Action: answer[X [Cite]]`.
    Those turns are still ASSISTANT turns, so grading them would teach the student to
    reproduce a format the environment rejects — training in a bug. The turn stays in
    the conversation as context (the model sees the mistake and the recovery), it just
    carries no loss.
    """
    nxt = history[i + 1] if i + 1 < len(history) else None
    return bool(nxt and nxt.get("role") == "user"
                and str(nxt.get("content", "")).startswith("error:"))


def encode_trajectory(tokenizer, history: list[dict], task_id: str = "",
                      max_len: int | None = None,
                      grade_rejected_turns: bool = False,
                      skip_final_answer: bool = False) -> SFTExample | None:
    """Encode one full conversation, grading ONLY assistant turns.

    `grade_rejected_turns=False` (the default) additionally SKIPS assistant turns the
    environment rejected — see `is_rejected_turn`. Set True only to study what the
    teacher did wrong, never to build a training set.

    `skip_final_answer=True` grades every turn EXCEPT the last assistant turn. Used for
    tier-B trajectories (the teacher searched and read the right passages but got the
    answer wrong): the process is worth imitating, the wrong answer is not. It is ~5
    tokens of ~170, but it is the highest-stakes span in the trajectory — training on it
    teaches a confident wrong fact WITH citations attached, which is the reward-hacking
    shape, and a policy that confidently emits one memorised wrong answer gives GRPO
    zero-variance groups and no gradient.

    `history` is `env._history`: alternating user/assistant dicts, where every `user`
    turn is either our opening prompt or a tool observation — both of which we must NOT
    train on.

    Returns None if the sequence exceeds `max_len`. It DROPS rather than truncates:
    truncating would cut the final answer turn, which is the one span we most need
    graded, and would silently teach the model never to emit the end-of-turn token.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    n_spans = 0
    mismatches = 0

    n_rejected_skipped = 0
    last_assistant = max((j for j, m in enumerate(history)
                          if m.get("role") == "assistant"), default=-1)
    for i, msg in enumerate(history):
        if msg.get("role") != "assistant":
            continue                      # extended below, when its turn is appended
        if not grade_rejected_turns and is_rejected_turn(history, i):
            n_rejected_skipped += 1
            continue                      # in context, but carries no loss
        if skip_final_answer and i == last_assistant:
            continue                      # tier B: keep the process, drop the answer

        prefix_ids = _ids(tokenizer.apply_chat_template(
            history[:i], add_generation_prompt=True, tokenize=True))
        full_ids = _ids(tokenizer.apply_chat_template(
            history[:i + 1], add_generation_prompt=False, tokenize=True))

        common = _common_prefix_len(prefix_ids, full_ids)
        if common != len(prefix_ids):
            # Template re-rendered an earlier turn. Fall back to the true common prefix
            # (never grade before it — over-grading is the dangerous direction) and
            # count it so the caller can see it happened.
            mismatches += 1
        start = min(common, len(full_ids))
        if start >= len(full_ids):
            continue                      # empty assistant turn — nothing to grade

        # Everything from the previous state up to `start` is context: mask it.
        labels.extend([IGNORE_INDEX] * (start - len(labels)))
        labels.extend(full_ids[start:])   # the assistant's own tokens, incl. end-of-turn
        input_ids = full_ids
        n_spans += 1

    if not input_ids or n_spans == 0:
        return None

    # The trailing user turn (if the conversation ends on a tool observation) is not
    # covered by the loop above; pad the mask out so labels lines up with input_ids.
    if len(labels) < len(input_ids):
        labels.extend([IGNORE_INDEX] * (len(input_ids) - len(labels)))
    assert len(labels) == len(input_ids), "labels/input_ids length mismatch"

    if max_len is not None and len(input_ids) > max_len:
        return None

    return SFTExample(input_ids=input_ids, labels=labels, task_id=task_id,
                      n_graded_spans=n_spans, prefix_mismatches=mismatches,
                      n_rejected_skipped=n_rejected_skipped)


def _norm(t: str) -> str:
    return " ".join((t or "").split())


def graded_runs(ex: "SFTExample") -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges that carry loss."""
    runs, start = [], None
    for i, x in enumerate(ex.labels):
        if x != IGNORE_INDEX and start is None:
            start = i
        elif x == IGNORE_INDEX and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(ex.labels)))
    return runs


def assert_masking_correct(tokenizer, ex: SFTExample, history: list[dict]) -> None:
    """Prove the mask is right instead of trusting it. Raises on any violation.

    STRUCTURAL, not textual. An earlier version searched the graded text for the literal
    phrase "search results:" and raised if it appeared — which fired on a perfectly good
    trajectory where the teacher wrote, in its own Thought, *"Both films have their
    countries of origin in the search results: ..."*. Flagging correct data as corrupt is
    its own kind of bug: it would have had us "fixing" a working collector, and a check
    that cries wolf gets switched off. (Found 2026-08-26 by running the validator over
    the first real batch, after Harpreet asked for the collected trajectories to be
    verified rather than assumed.)

    So instead of guessing at marker strings, decode each contiguous graded run and
    require it to BE one of the assistant turns. That is the actual invariant — no
    phrase can spoof it, and it holds regardless of what words appear in either side.

    Checked, each with a distinct silent failure mode:
      1. Something is graded at all (an all-masked example is inert and would quietly
         shrink the effective dataset).
      2. Every graded run decodes to text contained in an assistant turn. If ANY tool
         observation or prompt text is graded, this fails — that is the requirement, and
         it is what "we don't want to train on tool responses or our prompts" means.
      3. Graded runs never outnumber assistant turns.
      4. Nothing is graded before the first assistant turn begins.
    """
    if ex.n_graded_tokens == 0:
        raise AssertionError(f"{ex.task_id}: nothing graded — the example is inert")

    assistant_texts = [_norm(m.get("content", "")) for m in history
                       if m.get("role") == "assistant"]
    runs = graded_runs(ex)
    if len(runs) > len(assistant_texts):
        raise AssertionError(
            f"{ex.task_id}: {len(runs)} graded runs but only {len(assistant_texts)} "
            f"assistant turns — something outside the model's own output is graded")

    for a, b in runs:
        text = _norm(tokenizer.decode(ex.input_ids[a:b], skip_special_tokens=True))
        if not text:
            continue                       # a run of only special tokens (the eos) is fine
        # STRICT CONTAINMENT, one direction only. An earlier version also accepted
        # `at in text` (the graded run being a SUPERSET of an assistant turn) for
        # tokenizer-roundtrip slack — but that is precisely the dangerous direction: a
        # run extended backwards into the preceding tool observation still contains the
        # assistant turn, so it passed. Caught by the test that swallows tool output
        # into an existing run. Over-grading must fail; under-grading is merely wasteful.
        if not any(text in at for at in assistant_texts):
            raise AssertionError(
                f"{ex.task_id}: GRADED TEXT IS NOT MODEL OUTPUT — tokens [{a}:{b}] decode "
                f"to {text[:120]!r}, which matches no assistant turn. This is prompt or "
                f"tool-response text being trained on.")

    first_graded = runs[0][0]
    if first_graded == 0:
        raise AssertionError(
            f"{ex.task_id}: grading starts at token 0 — the opening prompt is being "
            f"trained on, not masked")


def summarize(examples: list[SFTExample]) -> dict:
    """Dataset-level stats. `graded_fraction` is the one to eyeball: for this task it
    should be smallish (tool observations are long, assistant turns are short). A value
    near 1.0 means the mask is inverted or absent — check before training, not after."""
    if not examples:
        return {"n": 0}
    gt = [e.n_graded_tokens for e in examples]
    tot = [len(e.input_ids) for e in examples]
    return {
        "n_examples": len(examples),
        "mean_seq_len": sum(tot) / len(tot),
        "max_seq_len": max(tot),
        "mean_graded_tokens": sum(gt) / len(gt),
        "mean_graded_fraction": sum(e.graded_fraction for e in examples) / len(examples),
        "mean_graded_spans": sum(e.n_graded_spans for e in examples) / len(examples),
        "total_prefix_mismatches": sum(e.prefix_mismatches for e in examples),
    }
