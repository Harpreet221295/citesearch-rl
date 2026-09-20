"""Offline proof that the SFT mask trains on the model's own tokens and nothing else.

Runs against the REAL Qwen tokenizer (no GPU, no network beyond the cached tokenizer),
because the whole boundary calculation is a property of the chat template — testing it
against a mock template would prove nothing about the thing we actually ship.

Masking bugs do not crash. The loss still falls, the run still finishes, and the model
learns to hallucinate search results. So these are assertions, not eyeballing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_LAB = Path(__file__).resolve().parent.parent.parent
for p in (str(_LAB), str(_LAB / "distill")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sft_data import (                                            # noqa: E402
    IGNORE_INDEX, encode_trajectory, assert_masking_correct, summarize,
)

MODEL = "Qwen/Qwen2.5-3B-Instruct"

# A realistic 2-hop trajectory in exactly env._history's shape: user turns are our
# opening prompt and our tool observations; assistant turns are the model's own text.
HISTORY = [
    {"role": "user", "content":
        "You are a research agent. Answer the question by SEARCHING a document corpus...\n"
        "Question: What nationality is the director of the film Blue Harvest?"},
    {"role": "assistant", "content":
        "Thought: First I need to find who directed Blue Harvest.\n"
        "Action: search[who directed the film Blue Harvest]"},
    {"role": "user", "content":
        "search results:\n"
        "[1] Blue Harvest (film): Blue Harvest is a 2009 drama film directed by Jane Doe.\n"
        "[2] Blue Harvest (album): a 2015 studio album by the Norwegian band Fjord."},
    {"role": "assistant", "content":
        "Thought: [1] names Jane Doe. Reading it to confirm before citing.\n"
        "Action: read[Blue Harvest (film)]"},
    {"role": "user", "content":
        "[Blue Harvest (film)]\n"
        "Blue Harvest is a 2009 drama film. It was directed by Jane Doe."},
    {"role": "assistant", "content":
        "Thought: Confirmed. Jane Doe is American per the director passage.\n"
        "Action: answer[American [Blue Harvest (film)] [Jane Doe (director)]]"},
]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def ex(tok):
    e = encode_trajectory(tok, HISTORY, task_id="fixture-2hop")
    assert e is not None
    return e


def test_lengths_line_up(ex):
    assert len(ex.input_ids) == len(ex.labels)


def test_one_graded_span_per_assistant_turn(ex):
    assert ex.n_graded_spans == 3


def test_template_is_prefix_stable(ex):
    """If this fails the chat template re-renders earlier turns, and every boundary in
    the dataset needs re-deriving. Loud on purpose."""
    assert ex.prefix_mismatches == 0


def test_opening_prompt_is_masked(ex):
    """Grading must not start at token 0 — that would train on our own instructions."""
    first = next(i for i, x in enumerate(ex.labels) if x != IGNORE_INDEX)
    assert first > 0


def test_tool_output_is_never_graded(tok, ex):
    """THE test. Tool observations must contribute zero loss, or we teach the model to
    invent retrieval results instead of retrieving them."""
    graded = tok.decode([t for t in ex.labels if t != IGNORE_INDEX],
                        skip_special_tokens=False)
    assert "search results:" not in graded
    assert "Norwegian band Fjord" not in graded       # distinctive tool-only text
    assert "2009 drama film" not in graded            # from the read observation


def test_every_assistant_turn_is_graded(tok, ex):
    """The complement: each assistant turn's distinctive text must appear in the graded
    span. Catches an over-eager mask that silently drops turns."""
    graded = tok.decode([t for t in ex.labels if t != IGNORE_INDEX],
                        skip_special_tokens=False)
    for needle in ("search[who directed the film Blue Harvest]",
                   "read[Blue Harvest (film)]",
                   "answer[American [Blue Harvest (film)] [Jane Doe (director)]]"):
        assert needle in graded, f"assistant text not graded: {needle!r}"


def test_end_of_turn_token_is_graded(tok, ex):
    """The stop signal must carry loss or the student never learns to stop."""
    graded = tok.decode([t for t in ex.labels if t != IGNORE_INDEX],
                        skip_special_tokens=False)
    assert tok.eos_token in graded


def test_graded_fraction_is_a_minority(ex):
    """Assistant turns are short, tool observations are long. A fraction near 1.0 means
    the mask is inverted or missing — a sanity bound, not a tuning target."""
    assert 0.0 < ex.graded_fraction < 0.6


def test_assert_masking_correct_passes(tok, ex):
    assert_masking_correct(tok, ex, HISTORY)


def test_assert_masking_correct_catches_an_inverted_mask(tok, ex):
    """Prove the checker actually fires — a check that never fails is not a check."""
    from sft_data import SFTExample
    bad = SFTExample(input_ids=list(ex.input_ids), labels=list(ex.input_ids),
                     task_id="deliberately-broken", n_graded_spans=3)
    with pytest.raises(AssertionError):
        assert_masking_correct(tok, bad, HISTORY)


# A real trajectory from the first collected batch (2wiki-train-43824). The teacher used
# the phrase "search results:" inside its OWN Thought. An earlier version of
# assert_masking_correct substring-matched that phrase and raised — flagging perfectly
# good data as corrupt. Kept as a regression test: a checker that cries wolf gets
# switched off, and it nearly had us "fixing" a working collector.
HISTORY_PHRASE_IN_THOUGHT = [
    {"role": "user", "content":
        "You are a research agent...\nQuestion: Are both M.M.M. 83 and Handy Andy "
        "(1934 film) from the same country?"},
    {"role": "assistant", "content":
        "Thought: I need to find the country of origin for both films.\n"
        "Action: search[M.M.M. 83 country]"},
    {"role": "user", "content":
        "search results:\n[1] M.M.M. 83: a 1966 Italian spy film.\n"
        "[2] Handy Andy (1934 film): a 1934 American comedy film."},
    {"role": "assistant", "content":
        "Thought: Both films have their countries of origin in the search results: "
        "M.M.M. 83 is Italian, Handy Andy is American. I will read both to confirm.\n"
        "Action: read[M.M.M. 83]"},
    {"role": "user", "content": "[M.M.M. 83]\nM.M.M. 83 is a 1966 Italian spy film."},
    {"role": "assistant", "content":
        "Thought: Confirmed Italian and American, so not the same country.\n"
        "Action: answer[no [M.M.M. 83] [Handy Andy (1934 film)]]"},
]


def test_phrase_from_tool_output_inside_a_thought_is_not_a_violation(tok):
    """The teacher may legitimately SAY "search results:" while reasoning. That is its
    own output and must be graded — the invariant is structural (is this text the
    model's?), not lexical (does it contain a scary phrase?)."""
    e = encode_trajectory(tok, HISTORY_PHRASE_IN_THOUGHT, task_id="2wiki-train-43824")
    assert e is not None
    assert_masking_correct(tok, e, HISTORY_PHRASE_IN_THOUGHT)     # must NOT raise
    graded = tok.decode([t for t in e.labels if t != IGNORE_INDEX],
                        skip_special_tokens=True)
    assert "in the search results:" in graded          # the Thought IS graded
    assert "1966 Italian spy film" not in graded       # the tool output is NOT


def test_checker_catches_a_graded_run_that_swallows_tool_output(tok):
    """The structural check must be at least as strict as the old lexical one where it
    actually mattered. Here a graded run is EXTENDED BACKWARDS into the preceding tool
    observation — the run count is unchanged, so only the decode-and-compare catches it.
    This is the path that proves the check is about content, not just arithmetic."""
    from sft_data import SFTExample
    e = encode_trajectory(tok, HISTORY, task_id="fixture")
    labels = list(e.labels)
    first = next(i for i, x in enumerate(labels) if x != IGNORE_INDEX)
    for i in range(max(0, first - 40), first):        # contiguous -> same run count
        labels[i] = e.input_ids[i]
    bad = SFTExample(input_ids=list(e.input_ids), labels=labels,
                     task_id="tool-output-swallowed", n_graded_spans=e.n_graded_spans)
    with pytest.raises(AssertionError, match="NOT MODEL OUTPUT"):
        assert_masking_correct(tok, bad, HISTORY)


def test_checker_catches_an_extra_graded_run(tok):
    """The other failure shape: an isolated graded island somewhere in the prompt or a
    tool observation, which shows up as more graded runs than assistant turns."""
    from sft_data import SFTExample
    e = encode_trajectory(tok, HISTORY, task_id="fixture")
    labels = list(e.labels)
    first = next(i for i, x in enumerate(labels) if x != IGNORE_INDEX)
    for i in range(max(0, first - 60), max(0, first - 20)):   # leaves a masked gap
        labels[i] = e.input_ids[i]
    bad = SFTExample(input_ids=list(e.input_ids), labels=labels,
                     task_id="extra-run", n_graded_spans=e.n_graded_spans)
    with pytest.raises(AssertionError, match="graded runs but only"):
        assert_masking_correct(tok, bad, HISTORY)


def test_max_len_drops_rather_than_truncates(tok):
    """Dropping keeps the final answer turn intact; truncating would cut exactly the
    span we most need graded."""
    assert encode_trajectory(tok, HISTORY, max_len=10) is None


def test_trajectory_with_no_assistant_turn_returns_none(tok):
    assert encode_trajectory(tok, [{"role": "user", "content": "hi"}]) is None


def test_summarize_shape(ex):
    s = summarize([ex])
    assert s["n_examples"] == 1
    assert s["total_prefix_mismatches"] == 0
    assert 0 < s["mean_graded_fraction"] < 0.6


# The environment rejects an unparseable action with "error: unknown or unparsed
# action...". Measured in the first collected batch (2026-08-26): 5 of 7 parse failures
# were the teacher writing `Answer: X [Cite]` instead of `Action: answer[X [Cite]]`.
# Those are still ASSISTANT turns, so grading them would train the student to reproduce
# a format its own environment rejects.
HISTORY_WITH_REJECTED_TURN = [
    {"role": "user", "content": "You are a research agent...\nQuestion: Who wrote it?"},
    {"role": "assistant", "content":
        "Thought: Find the author.\nAction: search[who wrote it]"},
    {"role": "user", "content": "search results:\n[1] Book: written by Kevin Smith."},
    {"role": "assistant", "content": "Thought: Confirming.\nAction: read[Book]"},
    {"role": "user", "content": "[Book]\nBook was written by Kevin Smith."},
    # THE BAD TURN — wrong wrapper, environment rejects it.
    {"role": "assistant", "content": "Answer: Kevin Smith [Book]"},
    {"role": "user", "content":
        "error: unknown or unparsed action. Use one of: search, read, answer."},
    # The teacher recovers.
    {"role": "assistant", "content":
        "Thought: Reformatting.\nAction: answer[Kevin Smith [Book]]"},
]


def test_rejected_turn_is_not_graded(tok):
    """The malformed turn must carry no loss, while still appearing as context so the
    model can learn the recovery."""
    e = encode_trajectory(tok, HISTORY_WITH_REJECTED_TURN, task_id="rejected")
    assert e is not None
    assert e.n_rejected_skipped == 1
    assert e.n_graded_spans == 3                     # 4 assistant turns, 1 skipped
    graded = tok.decode([t for t in e.labels if t != IGNORE_INDEX],
                        skip_special_tokens=True)
    assert "Answer: Kevin Smith" not in graded       # the mistake is NOT trained on
    assert "Action: answer[Kevin Smith [Book]]" in graded   # the recovery IS
    assert_masking_correct(tok, e, HISTORY_WITH_REJECTED_TURN)


def test_rejected_turn_can_be_graded_when_explicitly_asked(tok):
    """The escape hatch exists for studying teacher mistakes — never for training."""
    e = encode_trajectory(tok, HISTORY_WITH_REJECTED_TURN, task_id="rejected",
                          grade_rejected_turns=True)
    assert e.n_rejected_skipped == 0
    assert e.n_graded_spans == 4
