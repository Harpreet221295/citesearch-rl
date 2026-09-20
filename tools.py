"""The tool world — the 3 ReAct tools the deep-research agent acts through
(book §10.7). The search→read→answer loop, one tool per action:

  search(query)          -> top-k passage TITLES + snippets ranked by the retriever
  read(title)            -> the full text of one passage
  answer(text)           -> commit the final answer (terminal action)

Design mirrors finqa_agent/tools.py exactly (ToolSpec registry + a never-raising
executor) so the rollout loop and the veRL env treat both labs the same. The one
difference is the *world*: here it's a DocStore (corpus.py) instead of a TableWorld.

Why split search vs read (instead of one "retrieve" that dumps full passages)?
Because deep research is a *sequence* of decisions — query, skim titles/snippets,
decide which to open, maybe re-query — and we want the trajectory to expose each of
those so the reward can price efficiency (fewer, better-targeted reads) and the
eval can localize where the agent wins (better queries? better selection?). It also
keeps each observation small (snippets, then one passage) rather than flooding the
context, which matters for the masking (fewer retrieved tokens graded by accident).

Everything here is mechanics — safe scaffold. Nothing is a learning artifact:
`answer` just records the text; whether that answer is *correct/grounded* is the
reward's job (reward.py, TODO #1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from corpus import DocStore


class ToolError(Exception):
    """A real, typed failure from a tool. The message is what the agent sees, so
    make it specific and stable — the agent has to be able to recover from it."""


# ---------------------------------------------------------------------------
# The tools. Each takes (world, **args) and returns a result string, or raises
# ToolError. `search`/`read` also return the titles they surfaced (2nd element)
# so the rollout can record retrieved_titles for the hit-rate metric.
# ---------------------------------------------------------------------------
def _tool_search(world: DocStore, query: str, k: int = 3) -> tuple[str, list[str]]:
    q = str(query).strip()
    if not q:
        raise ToolError("search: empty query. Pass a natural-language query string.")
    hits = world.search(q, k=int(k))
    if not hits:
        return ("no results.", [])
    titles = [doc.title for doc, _ in hits]
    lines = [f"[{i+1}] {doc.title}: {doc.snippet()}" for i, (doc, _score) in enumerate(hits)]
    return ("search results:\n" + "\n".join(lines), titles)


def _tool_read(world: DocStore, title: str) -> tuple[str, list[str]]:
    doc = world.get(str(title))
    if doc is None:
        raise ToolError(
            f"read: no passage titled '{title}'. Use search(query) first and read an "
            f"exact title from its results.")
    return (f"[{doc.title}]\n{doc.text}", [doc.title])


def _tool_answer(world: DocStore, text: str) -> tuple[str, list[str]]:
    # Terminal action: just record the committed answer. Correctness/groundedness
    # is scored by the reward, never here.
    a = str(text).strip()
    if not a:
        raise ToolError("answer: empty answer. Provide the final answer text.")
    return (a, [])


@dataclass
class ToolSpec:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    fn: Any
    doc: str
    terminal: bool = False

    def render(self) -> str:
        args = list(self.required) + [f"{a}?" for a in self.optional]
        return f"- {self.name}({', '.join(args)}): {self.doc}"


TOOLS: dict[str, ToolSpec] = {
    "search": ToolSpec("search", ("query",), ("k",), _tool_search,
                       "search the corpus; returns the top-k passage titles + snippets"),
    "read":   ToolSpec("read", ("title",), (), _tool_read,
                       "read the full text of one passage by its exact title"),
    "answer": ToolSpec("answer", ("text",), (), _tool_answer,
                       "commit your final answer, citing the passages that support it as "
                       "[Passage Title]; ends the episode", terminal=True),
}


# ---------------------------------------------------------------------------
# Executor: run ONE parsed call against the world. Never raises — turns a
# ToolError (or an unexpected error) into a stable observation string so the
# rollout loop can record it and let the agent try to recover. Returns
# (observation, ok, error, retrieved_titles).
# ---------------------------------------------------------------------------
def execute(world: DocStore, name: str, args: dict[str, Any]) -> tuple[str, bool, str | None, list[str]]:
    spec = TOOLS.get(name)
    if spec is None:
        return (f"error: unknown tool '{name}'", False, "unknown_tool", [])

    missing = [a for a in spec.required if a not in args]
    if missing:
        return (f"error: {name} missing required arg(s): {', '.join(missing)}", False, "missing_arg", [])
    allowed = set(spec.required) | set(spec.optional)
    extra = [a for a in args if a not in allowed]
    if extra:
        return (f"error: {name} got unexpected arg(s): {', '.join(extra)}", False, "unexpected_arg", [])

    try:
        result, titles = spec.fn(world, **args)
        return (result, True, None, titles)
    except ToolError as e:
        return (f"error: {e}", False, "tool_error", [])
    except Exception as e:  # a bug in a tool or a bad arg type — still don't crash the run
        return (f"error: internal ({type(e).__name__}: {e})", False, "internal_error", [])


def render_toolset(names: list[str]) -> str:
    """The tool menu block for the agent's system prompt."""
    return "\n".join(TOOLS[n].render() for n in names if n in TOOLS)
