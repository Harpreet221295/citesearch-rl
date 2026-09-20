"""GPT-in-the-driver's-seat: an OpenAI model as the POLICY inside our real environment.

WHY THIS SHAPE (2026-08-26, Harpreet's call after the harness audit):
Across 320 rollouts of Qwen2.5-3B, exactly ZERO trajectories were both correct and
correctly cited (best citation-F1 among 19 correct answers: 0.667). Rejection-sampling
RFT from the base model's own rollouts therefore has nothing to mine. This is the
fallback `RFT_PLAN_AND_MODEL_DIAGNOSIS.md` already wrote down for exactly this case:
seed the SFT set from a stronger teacher instead.

THE ONE THING THAT WOULD BE EASY TO GET WRONG, and that this file exists to prevent:
do NOT ask GPT to *write* trajectories. It would invent search results that are not in
our corpus and we would fine-tune the student on fiction. Instead GPT is dropped in as
the policy of the SAME `DeepResearchEnv` the student uses: GPT chooses the action, OUR
tools answer from OUR corpus, `env.step()` does the bookkeeping. Every observation is
real and every trajectory is gradeable by the same `citations.py` / `metrics.py` the
student is scored with.

MODEL CHOICE — `gpt-4.1-mini`, deliberately not the strongest available.
Harpreet's constraint: "don't want to inject too much reasoning as well (too verbose
reasoning, I don't think our small model could do)". That is the right instinct and it
is the core constraint of distillation: the teacher's output distribution has to be
something a 3B student can actually imitate. The GPT-5.x family are *reasoning* models
that emit long hidden chains; imitating those with 3B produces a student that starts a
chain it cannot finish. `gpt-4.1-mini` is non-reasoning, strong at tool use, and cheap
($0.40/$1.60 per 1M in/out as of 2026-08-26). `_THOUGHT_BUDGET` below tightens it
further by instruction.

FORMAT — bracket/ReAct, not OpenAI function calling.
GPT would do better with native function calling, but `env.py` speaks bracket format and
GRPO will run in bracket format. Training the student on a format it will not be rolled
out in wastes the whole SFT stage. So GPT is asked to emit `Thought:/Action:` text and
is fed straight into `env.step()` — a drop-in policy swap, nothing else changes.

CONCURRENCY — threads across episodes, sequential within one.
The opposite of the vLLM path, and deliberately so. vLLM wants lockstep batching because
one engine serves all sequences; the OpenAI API is network-bound with per-request
latency, so the win is running many episodes at once. Turns inside an episode are
inherently sequential (turn N+1 depends on turn N's tool result).
"""
from __future__ import annotations

import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from env import DeepResearchEnv

# Non-reasoning, cheap, good at tools. See MODEL CHOICE above before changing this.
DEFAULT_MODEL = "gpt-4.1-mini"

# USD per 1M tokens, from developers.openai.com/api/docs/pricing (checked 2026-08-26).
# Only used for the running cost estimate; wrong numbers here cost nothing but a wrong
# estimate, so they are reported as an estimate and never as billing truth.
# Cached input tokens are billed at a fraction of the fresh rate. The only assumed
# number in the cost calculation — tokens themselves come from the API's own `usage`.
CACHED_DISCOUNT = 0.25

PRICING = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1":      (2.00, 8.00),
    "gpt-4o-mini":  (0.15, 0.60),
}

# Keep the teacher's reasoning short enough for a 3B student to imitate (see MODEL
# CHOICE). This is appended to the env's own opening prompt rather than replacing it,
# so the teacher sees EXACTLY the instructions the student sees, plus this one
# constraint — anything else would make the mined trajectories unrepresentative of the
# task the student is actually being trained for.
_THOUGHT_BUDGET = (
    "\nWrite your Thought as ONE short sentence, at most 20 words. Then the Action line. "
    "Nothing else: no preamble, no explanation, no restating the question."
    # READ-BEFORE-CITE. Found 2026-08-26 in the very first teacher smoke run, and it
    # explains far more than the teacher's own score.
    #
    # `citations.verify_citations` scores a citation only when the agent actually called
    # `read` on that passage: `tp_titles = distinct_cited & gold & read_titles`. That is
    # deliberate ("cite-what-you-read", see citations.py's docstring and NOTES.md), not a
    # bug — verified by hand before changing anything. In the smoke run the teacher cited
    # BOTH gold titles from search snippets alone and scored cite_f1 = 0.000; a second
    # episode read one of two and scored exactly 0.500.
    #
    # This is very likely the root cause of `groundedness ~= 0` across ALL FOUR reward
    # designs in TRAINING_HISTORY_LOG.md. Every healthy run converged on ~2 turns
    # (search -> answer, never read), and a citation without a read is unscoreable by
    # construction. Reward shaping could not fix it because the gate sits on a BEHAVIOUR
    # (read) the policy was not emitting — which is also why the 2026-08-24 discovery
    # that the prompt never demonstrated `read` matters so much.
    #
    # The student's own prompt ALREADY says "Do not cite a passage you did not read"; it
    # simply does not comply. So this instruction does not change the task or weaken the
    # metric — it makes the TEACHER obey the rule the student is already given, so the
    # mined trajectories demonstrate the exact behaviour that is missing.
    # ANSWER-LINE FORMAT. Measured in the first collected batch (2026-08-26): 5 of 7
    # parse failures were the teacher writing `Answer: Kevin Smith [Silent Bob Speaks]`
    # instead of `Action: answer[Kevin Smith [Silent Bob Speaks]]` — it drops the
    # `Action: answer[...]` wrapper on the FINAL turn only, having used `Action:`
    # correctly for every search and read. One episode never recovered, flip-flopping
    # between `Answer: ...` and a bare `answer[...]`, and scored wrong.
    # This costs more than a wasted turn: a failed turn is still an ASSISTANT turn, so it
    # would be graded, and the student would be trained to make the same mistake.
    "\nEvery turn, including the last, must use the exact form `Action: <tool>[<argument>]`. "
    "The final turn is `Action: answer[<short phrase> [Title] [Title]]` — never `Answer: ...`, "
    "never a bare `answer[...]` without the `Action: ` prefix. "
    # RESTORED 2026-08-26. This sentence was accidentally DELETED when the answer-format
    # instruction above was patched in — the edit replaced it instead of adding alongside
    # it. Measured cost: read_before_cite_rate fell 0.938 (pilot, sentence present) to
    # 0.79-0.84 (later batches, sentence gone). It is the single most load-bearing line in
    # this prompt, since cite-what-you-read is the exact behaviour SFT exists to install.
    "\nBefore you cite a passage you MUST have read it with read[...] in an earlier turn. "
    "Citing from a search snippet alone does not count. So: search to find candidate "
    "titles, then read[...] EVERY passage you intend to cite, then answer. Cite one "
    "passage per fact — a 2-hop question needs two citations, and each of those two "
    "passages must have been read."
)

# Same turn-boundary enforcement the student gets (diagnosis1._TURN_STOP_SEQUENCES).
# Without it the teacher free-runs and writes a whole hallucinated trajectory in one
# completion — the exact bug found in the student on 2026-08-25.
STOP = ["\nThought:", "\nsearch results:", "\n["]


@dataclass
class Usage:
    """Token/cost accounting, thread-safe. Reported as an ESTIMATE, never as billing."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0     # billed at a discount; see cost_usd
    calls: int = 0
    errors: int = 0
    model: str = DEFAULT_MODEL
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, p: int, c: int, cached: int = 0) -> None:
        with self._lock:
            self.prompt_tokens += p
            self.completion_tokens += c
            self.cached_prompt_tokens += cached
            self.calls += 1

    def add_error(self) -> None:
        with self._lock:
            self.errors += 1

    @property
    def cost_usd(self) -> float:
        """Measured tokens (from each response's `usage`, not estimated) x published
        price. Cached input is billed at a discount — this workload resends a long,
        identical prefix every turn, so ignoring it OVER-states cost, sometimes badly.
        CACHED_DISCOUNT is the published multiplier and is the one genuinely assumed
        number here; everything else is measured or fetched. Always an estimate, never
        billing truth."""
        pin, pout = PRICING.get(self.model, (0.0, 0.0))
        fresh = max(0, self.prompt_tokens - self.cached_prompt_tokens)
        return (fresh / 1e6 * pin
                + self.cached_prompt_tokens / 1e6 * pin * CACHED_DISCOUNT
                + self.completion_tokens / 1e6 * pout)

    def summary(self) -> str:
        cached = (f" ({self.cached_prompt_tokens:,} cached)"
                  if self.cached_prompt_tokens else "")
        return (f"{self.calls} calls, {self.prompt_tokens:,} in{cached} / "
                f"{self.completion_tokens:,} out tokens, ~${self.cost_usd:.3f} "
                f"({self.errors} errors)")


def make_client():
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not found — add it to "
                         f"{_HERE/'.env'} (gitignored).")
    from openai import OpenAI
    return OpenAI(api_key=key)


def _complete(client, model, messages, usage: Usage, temperature: float,
              max_tokens: int, max_retries: int = 5) -> tuple[str, bool]:
    """One assistant turn, with backoff. Returns (text, gave_up).

    Returns "" rather than raising, so one bad episode cannot kill a whole collection
    run — the env treats an empty turn as a parse failure. But `gave_up` is now reported
    so the caller can EXCLUDE that episode from training data: a turn that is empty
    because the API rate-limited us is not a demonstration of anything, and silently
    training on it would teach the model to emit nothing. Added 2026-08-26 after a
    16-worker run produced 60 transient errors in 268 calls and 3 empty answers in 80
    episodes.
    """
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, stop=STOP,
            )
            if r.usage:
                det = getattr(r.usage, "prompt_tokens_details", None)
                usage.add(r.usage.prompt_tokens, r.usage.completion_tokens,
                          int(getattr(det, "cached_tokens", 0) or 0))
            return (r.choices[0].message.content or ""), False
        except Exception as e:                                    # noqa: BLE001
            usage.add_error()
            transient = any(s in type(e).__name__.lower()
                            for s in ("ratelimit", "timeout", "apiconnection",
                                      "internalserver", "apistatus"))
            if attempt == max_retries - 1 or not transient:
                if not transient:
                    print(f"    [teacher] non-transient error: {type(e).__name__}: {e}")
                return "", True
            time.sleep(min(2 ** attempt + random.random(), 30))
    return "", True


def run_episode(client, task, cfg, usage: Usage, model: str = DEFAULT_MODEL,
                temperature: float = 0.3, max_tokens: int = 220):
    """Drive ONE episode of the real env with GPT as the policy.

    Returns (trajectory, raw_assistant_turns, history).

    `history` is `env._history` — the FULL conversation, and the thing
    `sft_data.encode_trajectory` consumes. Returning it is not optional: an earlier
    version returned only (traj, raws), `collect.py` tried to recover the conversation
    via `getattr(traj, "_history", [])`, and Trajectory has no such field — so every
    collected record silently stored `history: null` and the whole collection would have
    been unusable for SFT. Found 2026-08-26 only because Harpreet asked to go
    incrementally instead of generating everything in one run.

    The raw turns are kept too, because production `env.py` drops the model's text on a
    parse failure (env.py:122) — the gap that hid 257 rejected replies from the
    student-side analysis. Not repeating that here.

    `temperature=0.3` by default, not GRPO's 0.9: we are mining the teacher's BEST
    behaviour to imitate, not sampling its distribution. Raise it only if a diversity
    problem shows up in the mined set (the plan doc's data-hygiene section).
    """
    env = DeepResearchEnv.from_dict({"task": task, "cfg": cfg, "judge": None})
    obs = env.reset()
    # env.reset() seeds _history with the opening prompt as a user turn. Append the
    # brevity constraint to THAT message so the teacher sees the student's exact
    # instructions plus this one addition, and nothing else differs.
    if env._history and env._history[0].get("role") == "user":
        env._history[0]["content"] += _THOUGHT_BUDGET

    raw_turns: list[str] = []
    api_gave_up = False
    for _ in range(int(getattr(cfg, "max_turns", 8)) + 1):
        msgs = list(env._history)
        text, gave_up = _complete(client, model, msgs, usage, temperature, max_tokens)
        api_gave_up = api_gave_up or gave_up
        raw_turns.append(text)
        _obs, _r, done, _info = env.step(text)
        if done:
            break
    return (env._to_dr_trajectory(), raw_turns,
            [dict(m) for m in env._history], api_gave_up)


def collect(client, tasks, cfg, usage: Usage, model: str = DEFAULT_MODEL,
            temperature: float = 0.3, k: int = 1, workers: int = 8,
            progress_every: int = 10, max_cost_usd: float | None = None,
            on_episode=None):
    """Run `k` episodes per task, `workers` in flight. Returns (task, traj, raws, history).

    Submitted in WAVES rather than all at once, so `max_cost_usd` can actually halt the
    run part-way. A cap that is only checked after every job has been submitted is not a
    cap. Harpreet's instruction (2026-08-26): go step by step, fix issues as they appear,
    do not burn money on one big run.

    `on_episode(task, traj, raws, history, api_gave_up)` fires as each episode lands — use it to
    append to disk immediately, so an interrupted run keeps everything it already paid
    for.
    """
    jobs = [t for t in tasks for _ in range(k)]
    out, done_n = [], 0
    t0 = time.time()
    wave = max(1, workers * 2)
    stopped = False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in range(0, len(jobs), wave):
            if max_cost_usd is not None and usage.cost_usd >= max_cost_usd:
                print(f"    [teacher] COST CAP reached (${usage.cost_usd:.3f} >= "
                      f"${max_cost_usd:.2f}) — stopping after {done_n}/{len(jobs)} "
                      f"episodes. Everything already collected is saved.")
                stopped = True
                break
            chunk = jobs[start:start + wave]
            futs = {ex.submit(run_episode, client, t, cfg, usage, model, temperature): t
                    for t in chunk}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    traj, raws, history, gave_up = fut.result()
                    out.append((task, traj, raws, history, gave_up))
                    if on_episode is not None:
                        on_episode(task, traj, raws, history, gave_up)
                except Exception as e:                            # noqa: BLE001
                    usage.add_error()
                    print(f"    [teacher] episode failed for {task.task_id}: "
                          f"{type(e).__name__}: {e}")
                done_n += 1
                if done_n % progress_every == 0 or done_n == len(jobs):
                    print(f"    [teacher] {done_n}/{len(jobs)} episodes  "
                          f"{time.time()-t0:.0f}s  {usage.summary()}", flush=True)
    return out, stopped
