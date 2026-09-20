"""The three worked examples from `diagnosis1._RICH_EXAMPLES`, translated into native
tool-calling CONVERSATION TURNS.

WHY: the 2026-08-25 bracket-vs-native A/B was never apples-to-apples. The bracket arm
carried three rich worked trajectories in its prompt; the native arm carried none (by
design — native calling is meant to need little in-context teaching). So bracket's
correctness lead (0.188 vs 0.047 on the fixed harness) cannot be attributed to SYNTAX;
it could just as easily be the demonstrations. This module fills the missing 2x2 cell.

FAITHFULNESS RULES followed here, so the comparison isolates format and nothing else:
  * Same three questions, same hop counts, same search results, same passage text, same
    reasoning sentences, same final answers and citations as `_RICH_EXAMPLES`. Nothing
    was improved, shortened, or reworded while translating.
  * Tool responses reproduce `tools.py`'s real output format verbatim (search results as
    "search results:\\n[1] Title: snippet", reads as "[Title]\\n<full text>"), exactly as
    the bracket examples do — the model must see what it will really receive.
  * The per-step reasoning rides in the assistant message's `content`, alongside
    `tool_calls`. That is the native analogue of ReAct's `Thought:` line, and Qwen's
    template renders it ({%- if message.content %}). NOTE this means using these examples
    implicitly demonstrates reasoning too — which is correct, because the bracket arm's
    examples demonstrate `Thought:` as well. Matching it is the point; keeping the arms
    different again would defeat the exercise.

ONE DELIBERATE DIFFERENCE, and the reason for it:
The bracket examples end with a final `Thought:` and then `answer[...]`. Here the final
assistant message is the answer phrase ALONE, with no preceding prose. It has to be: in
the native convention a plain-text turn IS the final answer, so a closing thought would
be swallowed into the answer string and scored as part of it. Demonstrating a terse
final turn is what we actually want imitated.

Few-shot examples are given as real prior conversation turns rather than as text pasted
into the system prompt, because that is how an instruction-tuned chat model was trained
to consume them.
"""
from __future__ import annotations


def _search(query: str) -> dict:
    return {"role": "assistant", "tool_calls": [
        {"function": {"name": "search", "arguments": {"query": query, "k": 3}}}]}


def _read(title: str) -> dict:
    return {"role": "assistant", "tool_calls": [
        {"function": {"name": "read", "arguments": {"title": title}}}]}


def _thinking(thought: str, call: dict) -> dict:
    """Attach the step's reasoning to the assistant turn that makes the call — the
    native analogue of ReAct's `Thought:` preceding its `Action:`."""
    return {**call, "content": thought}


# --------------------------------------------------------------------------- #
# Example 1 — 2-hop, clean.
# --------------------------------------------------------------------------- #
_EX1 = [
    {"role": "user", "content": "Question: What nationality is the director of the film Blue Harvest?"},
    _thinking("First I need to find who directed Blue Harvest.",
              _search("who directed the film Blue Harvest")),
    {"role": "tool", "content":
        "search results:\n"
        "[1] Blue Harvest (film): Blue Harvest is a 2009 drama film. It was directed by Jane Doe and premiered at the Toronto festival. The film follows a family farm over one summer.\n"
        "[2] Blue Harvest (album): Blue Harvest is a 2015 studio album by the Norwegian band Fjord. It reached number three on the domestic chart.\n"
        "[3] John Roe (producer): John Roe is a British film producer who has worked on several independent dramas."},
    _thinking("[1] is the film, and it names Jane Doe as director. Reading it to confirm before citing.",
              _read("Blue Harvest (film)")),
    {"role": "tool", "content":
        "[Blue Harvest (film)]\n"
        "Blue Harvest is a 2009 drama film. It was directed by Jane Doe and premiered at the Toronto festival. The film follows a family farm over one summer."},
    _thinking("Confirmed — Jane Doe directed it. Now I need her nationality.",
              _search("Jane Doe director nationality")),
    {"role": "tool", "content":
        "search results:\n"
        "[1] Jane Doe (director): Jane Doe is an American film director, born in Ohio in 1971. She is known for independent dramas including Blue Harvest and Winter Wheat.\n"
        "[2] Blue Harvest (film): Blue Harvest is a 2009 drama film. It was directed by Jane Doe and premiered at the Toronto festival.\n"
        "[3] John Roe (producer): John Roe is a British film producer who has worked on several independent dramas."},
    _thinking("[1] says she is American. Reading it to confirm before citing.",
              _read("Jane Doe (director)")),
    {"role": "tool", "content":
        "[Jane Doe (director)]\n"
        "Jane Doe is an American film director, born in Ohio in 1971. She is known for independent dramas including Blue Harvest and Winter Wheat."},
    {"role": "assistant", "content": "American [Blue Harvest (film)] [Jane Doe (director)]"},
]

# --------------------------------------------------------------------------- #
# Example 2 — 3-hop, and reading the FULL passage changes the answer the snippet
# alone would have given.
# --------------------------------------------------------------------------- #
_EX2 = [
    {"role": "user", "content":
        "Question: In what year did the founder of the company that manufactures the 'Halcyon Drive' bicycle first found a company?"},
    _thinking("First I need to find who manufactures the Halcyon Drive bicycle.",
              _search("manufacturer of Halcyon Drive bicycle")),
    {"role": "tool", "content":
        "search results:\n"
        "[1] Halcyon Drive: Halcyon Drive is a carbon-frame touring bicycle manufactured by Ridgeline Cycles, released in 2018 and marketed toward long-distance riders.\n"
        "[2] Halcyon (mythology): In Greek mythology, the halcyon is a bird associated with calm seas during the winter solstice.\n"
        "[3] Meridian Bikes: Meridian Bikes is an unrelated bicycle brand based in Denver, known for mountain bikes."},
    _thinking("[1] says it's made by Ridgeline Cycles. Reading to confirm before moving on.",
              _read("Halcyon Drive")),
    {"role": "tool", "content":
        "[Halcyon Drive]\n"
        "Halcyon Drive is a carbon-frame touring bicycle manufactured by Ridgeline Cycles, released in 2018 and marketed toward long-distance riders."},
    _thinking("Confirmed — Ridgeline Cycles. Now I need who founded Ridgeline Cycles.",
              _search("founder of Ridgeline Cycles")),
    {"role": "tool", "content":
        "search results:\n"
        "[1] Ridgeline Cycles: Ridgeline Cycles is a bicycle manufacturer headquartered in Portland, founded in 2010 by Tom Ashford. The company specializes in touring and gravel bikes, including the Halcyon Drive mod…\n"
        "[2] Tom Ashford (engineer): Tom Ashford is an American mechanical engineer known for frame-design patents filed in the early 2000s.\n"
        "[3] Portland Cycling Co-op: A nonprofit organization supporting cyclists in Portland, unrelated to Ridgeline Cycles."},
    _thinking("[1] names Tom Ashford as founder, in 2010. But the question asks when he FIRST "
              "founded a company — the snippet is truncated ('…'), so 2010 might just be when "
              "Ridgeline itself started, not his first company. Reading the FULL passage to check, "
              "not just trusting the snippet.",
              _read("Ridgeline Cycles")),
    {"role": "tool", "content":
        "[Ridgeline Cycles]\n"
        "Ridgeline Cycles is a bicycle manufacturer headquartered in Portland, founded in 2010 by Tom Ashford. The company specializes in touring and gravel bikes, including the Halcyon Drive model. Before starting Ridgeline, Ashford had founded Ashford Metalworks in 2003, a small custom frame-building shop that later closed when he shifted focus to full bicycle manufacturing."},
    {"role": "assistant", "content": "2003 [Halcyon Drive] [Ridgeline Cycles]"},
]

# --------------------------------------------------------------------------- #
# Example 3 — the first search misses and must be reformulated; also a
# same-name-different-person distractor.
# --------------------------------------------------------------------------- #
_EX3 = [
    {"role": "user", "content":
        "Question: What continent is the birthplace city of the author who wrote 'Coral Bell Testament' located on?"},
    _thinking("I need to find who wrote 'Coral Bell Testament'.",
              _search("author of Coral Bell Testament")),
    {"role": "tool", "content":
        "search results:\n"
        "[1] Coral Bell Testament (album): Coral Bell Testament is a 2016 concept album by the band Wintervale, exploring themes of memory and loss.\n"
        "[2] Testament Records: An independent record label based in Bristol, known for folk and ambient releases.\n"
        "[3] Coral Reef Conservation Trust: A nonprofit focused on marine ecosystem preservation, unrelated to the album or novel."},
    _thinking("None of these name an author of a NOVEL — [1] is an album, [2] and [3] are "
              "unrelated. My query was too generic. Searching more specifically for the novel.",
              _search("novel Coral Bell Testament writer")),
    {"role": "tool", "content":
        "search results:\n"
        "[1] Coral Bell Testament (novel): Coral Bell Testament is a 2011 novel written by Priya Anand, following three generations of a coastal family through decades of change.\n"
        "[2] Coral Bell Testament (album): Coral Bell Testament is a 2016 concept album by the band Wintervale, exploring themes of memory and loss.\n"
        "[3] Priya Anand (poet): Priya Anand is an unrelated poet based in Mumbai, known for a 2019 collection of verse."},
    _thinking("[1] is the novel, naming Priya Anand as author. Note [3] is a DIFFERENT Priya "
              "Anand (a poet) — I need the novelist, not her. Reading [1] to confirm before moving on.",
              _read("Coral Bell Testament (novel)")),
    {"role": "tool", "content":
        "[Coral Bell Testament (novel)]\n"
        "Coral Bell Testament is a 2011 novel written by Priya Anand, following three generations of a coastal family through decades of change."},
    _thinking("Confirmed — Priya Anand (the novelist). Now I need her birthplace.",
              _search("Priya Anand novelist birthplace")),
    {"role": "tool", "content":
        "search results:\n"
        "[1] Priya Anand (author): Priya Anand is a novelist born in Accra. She has written several works exploring family and migration, including Coral Bell Testament.\n"
        "[2] Priya Anand (poet): Priya Anand is an unrelated poet based in Mumbai, known for a 2019 collection of verse.\n"
        "[3] Accra: Accra is the capital and largest city of Ghana, located on the Gulf of Guinea coast."},
    _thinking("[1] is the right Priya Anand (matches the novel) and says she was born in Accra. "
              "Reading to confirm before citing.",
              _read("Priya Anand (author)")),
    {"role": "tool", "content":
        "[Priya Anand (author)]\n"
        "Priya Anand is a novelist born in Accra. She has written several works exploring family and migration, including Coral Bell Testament."},
    _thinking("Confirmed — born in Accra. I should ground the CONTINENT in a passage too, not "
              "just assume it. Reading the Accra passage from the search results.",
              _read("Accra")),
    {"role": "tool", "content":
        "[Accra]\n"
        "Accra is the capital and largest city of Ghana, located on the Gulf of Guinea coast in West Africa."},
    {"role": "assistant", "content":
        "Africa [Coral Bell Testament (novel)] [Priya Anand (author)] [Accra]"},
]


def worked_examples() -> list[dict]:
    """The three examples as prior conversation turns, ready to prepend to a history."""
    return [*_EX1, *_EX2, *_EX3]
