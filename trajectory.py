"""Shared data types for the deep_research_agent GRPO pipeline.

A *trajectory* is one on-policy attempt by the agent at one multi-hop question: a
ReAct sequence of (thought -> tool call -> real observation) turns — search the
corpus, read a passage, maybe search again — ending in a final answer or a forced
stop at max_turns.

Two boundaries are encoded here at the type level, and both are load-bearing
(same design as finqa_agent — this is a deliberate port):

  1. AGENT vs ENV, per Step:
       thought / call  -> the AGENT produced these
       observation     -> the ENVIRONMENT (retriever / doc store) produced this
                          i.e. the retrieved passages — the <information> block.
     This is what the reward reasons over.

  2. MODEL vs TOOL, per TOKEN, carried on the whole trajectory:
       input_ids   -> the exact token ids of the full ReAct sequence
                      (prompt + every assistant turn + every retrieved-passage block)
       model_mask  -> 1 where the MODEL emitted the token, 0 where the ENV/template
                      injected it (the prompt, role headers, every retrieved passage).
     This is the mask the GRPO loss must sum over — for BOTH the policy-gradient
     term AND the KL term. Grading a retrieved-passage token (marking it 1) is the
     L4 silent bug: the model gets rewarded/penalized for text it copied, not wrote.
     `env.assert_verl_masking_matches` (TODO) exists to catch exactly that once veRL
     owns the tokenization.

Deep-research specifics vs finqa: the observation here is *retrieved text* (long,
model didn't write it) rather than a small table cell — so the masking stakes are
higher (more tool tokens to accidentally grade), and `supporting_titles` lets us
score whether the agent actually retrieved the gold evidence (retrieval hit-rate).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class ToolCall:
    """One well-formed intended call. `raw` keeps the exact text the model emitted
    (before parsing) so the reward's format check can judge the real output."""
    name: str                      # "search" | "read" | "answer"
    args: dict[str, Any]
    raw: str = ""


@dataclass
class Step:
    """One ReAct turn.

    thought      : assistant reasoning text                 (AGENT)
    call         : the parsed tool call, or None            (AGENT, if any)
    observation  : the tool router's real reply text        (ENV) — retrieved passages
    ok           : did the tool run without error?
    error        : the typed error string, if any
    parse_ok     : did the model's Action line parse into a legal call at all?
    retrieved_titles : for a `search`/`read`, which corpus doc titles came back
                       (used by the retrieval-hit-rate metric; ENV-produced).
    """
    thought: str
    call: ToolCall | None
    observation: str
    ok: bool
    error: str | None = None
    parse_ok: bool = True
    retrieved_titles: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    """One full on-policy attempt at one multi-hop question."""
    task_id: str
    query: str
    tools: list[str]
    gold_answer: str                              # the short-answer gold (EM/F1 target)
    gold_aliases: list[str] = field(default_factory=list)   # acceptable answer variants
    supporting_titles: list[str] = field(default_factory=list)  # gold evidence paragraph titles
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None               # text after the answer tool if the agent finished
    done: bool = False                            # emitted an answer (vs hit max_turns)?

    # --- on-policy tokenization (filled by the rollout/env; consumed by the loss) ---
    input_ids: list[int] = field(default_factory=list)
    model_mask: list[int] = field(default_factory=list)   # 1 = model token, 0 = env/template

    # --- filled by reward.py / trainer (kept so a saved traj is self-describing) ---
    reward: float | None = None
    correct: bool | None = None                   # outcome correctness (EM against gold/aliases)
    f1: float | None = None                       # token-F1 of final answer vs gold
    judge_score: float | None = None              # groundedness judge score in [0, 1]
    meta: dict[str, Any] = field(default_factory=dict)   # seed, dataset, group_id, corpus size…

    # --- convenience views (diagnostics / localized eval metrics) ---
    @property
    def n_turns(self) -> int:
        return len(self.steps)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.call for s in self.steps if s.call is not None]

    @property
    def n_searches(self) -> int:
        return sum(1 for c in self.tool_calls if c.name == "search")

    @property
    def retrieved_title_set(self) -> set[str]:
        """Every corpus title the agent surfaced across all turns — the denominator
        for retrieval precision and the pool the hit-rate checks against."""
        out: set[str] = set()
        for s in self.steps:
            out.update(s.retrieved_titles)
        return out

    def retrieval_hit_rate(self) -> float:
        """Fraction of gold supporting titles the agent actually retrieved. A localized
        signal the capstone asks for — did the agent find the evidence, regardless of
        whether it got the final answer right? 1.0 if there are no gold titles recorded."""
        gold = set(self.supporting_titles)
        if not gold:
            return 1.0
        found = gold & self.retrieved_title_set
        return len(found) / len(gold)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        steps = [
            Step(
                thought=s["thought"],
                call=(ToolCall(**s["call"]) if s["call"] else None),
                observation=s["observation"],
                ok=s["ok"],
                error=s.get("error"),
                parse_ok=s.get("parse_ok", True),
                retrieved_titles=s.get("retrieved_titles", []),
            )
            for s in d.get("steps", [])
        ]
        d = {**d, "steps": steps}
        return Trajectory(**d)


def dump_trajectories(path: str, trajs: list[Trajectory]) -> None:
    with open(path, "w") as f:
        for t in trajs:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_trajectories(path: str) -> list[Trajectory]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Trajectory.from_dict(json.loads(line)))
    return out
