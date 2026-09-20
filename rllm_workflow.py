"""rllm_workflow.py — the rLLM Workflow adapter for DeepResearchEnv (HANDOFF.md step 2).

Wraps the already-built, already-tested DeepResearchEnv (env.py) + reward (reward.py)
in rLLM's real Workflow/BaseAgent contract — confirmed against the INSTALLED
rllm@9beb6e0 source, not docs (RLLM_VERL_INSTALL_NOTES.md's hard-won rule: docs pages
disagreed with each other and with reality three separate times this build; source
didn't). rLLM's own built-in `MultiTurnWorkflow` looks like a drop-in fit (same
agent_cls/env_cls/max_steps shape) but its `run()` discards the ModelOutput after
`response = output.text` — which silently breaks masking (see below) — so this is a
small custom Workflow, not a reuse of MultiTurnWorkflow verbatim.

THE MASKING MECHANISM — confirmed by reading rllm/trainer/verl/transform.py::
_process_trajectory, not guessed: veRL does NOT need a hand-built token-level mask at
all anymore (env.assert_verl_masking_matches's "roll a rollout, decode where mask==0/1,
assert the passage text is in the ignored span" design assumed the OLD API where you'd
build the mask yourself and had to verify it — that model is obsolete for this API,
see the note in env.py once wired). Instead: transform.py walks `trajectory.steps`,
and for each step needs `step.model_output.prompt_ids` / `.completion_ids` (the exact
tokens the rollout engine actually tokenized-and-generated — WE never tokenize
anything ourselves). If step N's `prompt_ids` is a prefix-extension of step N-1's
(`prompt_ids + completion_ids`), the DELTA between them is auto-masked 0 (it's an
"observation" — the tool text / injected context inserted between turns) and each
step's own `completion_ids` stays masked 1 (the model's own generation). This is a
cumulative-prefix merge, correct BY CONSTRUCTION as long as every Step is built via
`Step.from_model_output(model_output, messages=..., action=...)` (the sanctioned
factory in rllm/types.py) using the REAL ModelOutput the rollout engine returned.
Concretely: retrieved-passage text only ever enters the trajectory as a `role=user`
message between two `role=assistant` turns — never as part of any step's
`completion_ids` — so it always lands in the auto-masked delta, never the graded span.
Worth eyeballing once wired anyway (train_dr.py's `--mask-check`) — same "trust but
verify" instinct this whole build runs on, just checking a DIFFERENT mechanism now.

Dataset contract (the sibling `# VERIFY` from HANDOFF §2 item 3, resolved alongside
this): confirmed via `rllm/engine/agent_workflow_engine.py::execute_tasks_verl` —
`tasks = batch.non_tensor_batch["extra_info"].tolist()`. The `task` dict this
Workflow's `run()`/`reset()` receives is EXACTLY the `extra_info` column of whatever
parquet file `data.train_files`/`data.val_files` point at (verl's own `RLHFDataset`
convention — also needs a `prompt` column present, even though we never use it: our
Workflow builds its own opening ReAct prompt via `env._opening_prompt`, not verl's
prompt-templating path). See `train_dr.write_verl_dataset` for the writer.

THE BATCH-DIVISIBILITY BUG (2026-08-23) — a real blocker, found + root-caused + patched
this session, full story in WORKFLOW_PORT_NOTES.md bug 11/12. Short version: veRL merges
a trajectory's steps into ONE training row only when they form a clean token-level
cumulative-prefix chain (see masking note above); when they don't, the trajectory
contributes >1 row, making the total row count per training step NOT reliably equal to
`train_batch_size × group_size`. CONFIRMED (via `_diagnose_cumulative_break` below,
finish_reason='stop' on every break) this is NOT about `max_new_tokens` truncation — the
leading remaining hypothesis is BPE tokenizer non-prefix-stability when the rollout engine
retokenizes the full growing conversation from scratch each turn (a known challenge class
for multi-turn chat-template tokenization, not specific to our code). MEASURED severity:
negligible at sanity's `group_size=2` (divisor 2, ~50/50 by chance) but a near-CERTAIN
crash at cloud's `group_size=16` (divisor 16; one real test hit `42 % 16 != 0` on the
first attempt, ~31% of trajectories breaking). `_patch_make_iterator_for_ragged_batches`
below is the fix — applied at import time, see its own docstring for why it's safe.
"""
from __future__ import annotations

from rllm.agents.agent import BaseAgent
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.types import Action, Episode, Step, Trajectory
from rllm.workflows.timing_mixin import TimingTrackingMixin
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

from data import DRTask
from env import DeepResearchEnv


def _patch_make_iterator_for_ragged_batches() -> None:
    """PATCH (2026-08-23) for the batch-divisibility bug — see module docstring.

    `verl.utils.tensordict_utils.make_iterator` hard-asserts
    `batch_size % mini_batch_size == 0` — CONFIRMED (web-searched) a deliberate verl
    design choice, not a bug: verl's SPMD strategy needs every data-parallel worker
    to see an identically-shaped mini-batch. But the actual collected batch size is
    NOT reliably predictable in advance for a multi-turn agent whose trajectories can
    occasionally split (see above) — so a fixed a-priori `mini_batch_size` can't be
    guaranteed to divide it, and at cloud-scale `group_size` this crashes almost every
    step (measured, not assumed — see module docstring).

    Fix: when the requested `mini_batch_size` doesn't evenly divide the actual
    collected batch, fall back to ONE full-batch mini-batch (`mini_batch_size =
    actual_size`) instead of crashing. **Why this is safe, not a correctness
    compromise**: `mini_batch_size` only controls how many PPO mini-batch gradient
    steps happen per epoch over an ALREADY-collected, already-correct batch — using
    one large mini-batch instead of several smaller ones is a legitimate (if less
    sample-efficient) PPO/GRPO configuration choice, not a violation of SPMD (every
    data-parallel worker still sees the identical, well-formed whole batch) and does
    not touch gradient correctness, masking, or advantage computation at all — purely
    how the collected batch gets chunked for the optimizer loop. `engine_workers.py`
    calls this via `tu.make_iterator(...)` (module-attribute access, not a pre-bound
    `from ... import`), so patching the attribute on the module object is sufficient
    — confirmed by reading the call site, not assumed.

    ⚠️ CORRECTED 2026-08-23 (an earlier version of this docstring claimed "applied
    once, at import time, in every process that imports this module" — that was
    WRONG, disproven empirically: `train_mini_batch`/`make_iterator` actually run
    inside `WorkerDict`/`ActorRolloutRefWorker`, a SEPARATE Ray actor spawned by
    verl's own worker-group machinery (`ray.remote(actor_rollout_cls)` in
    `TaskRunner.add_actor_rollout_worker`) that NEVER imports `rllm_workflow.py` at
    all — same class of process-boundary issue as the global-step investigation
    (HANDOFF.md item 5), just hit again in a new spot. A verification run with this
    patch "applied" still crashed with the exact same assertion, patch never firing
    — that's what caught it, not a review). **The real fix**: this function must be
    registered as a Ray `worker_process_setup_hook` (see `_ray_worker_setup_hook`
    below + `train_dr.py`'s `main()`, which calls `ray.init()` itself with this
    hook merged into verl's own runtime_env BEFORE `AgentTrainer` gets a chance to
    — its own `ray.init()` call is guarded by `if not ray.is_initialized()`, so
    ours wins). That hook runs in EVERY new Ray worker process at startup,
    regardless of what it's about to execute — the only way to reliably reach a
    framework-owned worker process we don't control the imports of."""
    import verl.utils.tensordict_utils as tu

    if getattr(tu.make_iterator, "_dr_patched", False):
        return
    _orig_make_iterator = tu.make_iterator

    def _patched(tensordict, mini_batch_size, *args, **kwargs):
        actual = tensordict.batch_size[0]
        if mini_batch_size and actual % mini_batch_size != 0:
            print(
                f"[batch-safety-patch] {actual} collected rows not divisible by "
                f"requested mini_batch_size={mini_batch_size} (a trajectory's "
                f"cumulative-prefix merge produced extra rows — see "
                f"WORKFLOW_PORT_NOTES.md bug 11/12). Falling back to one full-batch "
                f"mini-batch ({actual}) instead of crashing."
            )
            mini_batch_size = actual
        return _orig_make_iterator(tensordict, mini_batch_size, *args, **kwargs)

    _patched._dr_patched = True
    tu.make_iterator = _patched


_TRAINING_STEP_STATE = {"step": 0}


def _patch_agent_workflow_engine_step_tracking() -> None:
    """RESOLVED 2026-08-24 — the `lambda_eff` ramped-efficiency-toll wiring gap noted
    as "STILL OPEN" in HANDOFF.md/WORKFLOW_PORT_NOTES.md (an earlier investigation
    concluded no built-in hook existed and a real fix would need a Ray-actor-level
    monkeypatch). That conclusion was WRONG — missed on the first pass, found by
    actually reading `rllm/trainer/verl/agent_workflow_trainer.py` this time:
    `AgentWorkflowPPOTrainer.fit()` calls
    `self.agent_execution_engine.set_training_step(self.global_steps, mode=..., epoch=...)`
    every iteration (train AND periodic-eval passes) — a real, existing, intended hook.
    Its ONLY built-in consumer is episode-logging (`AgentWorkflowEngine.current_step`,
    read by the episode logger, never passed into `workflow.run()`/`reset()`'s own
    kwargs) — so it doesn't reach our Workflow instances by itself, but it IS the
    correct step number, updated at the right time, in the right process.

    Confirmed same-process, not a cross-actor problem the way the batch-safety-patch
    was: `agent_execution_engine = AgentWorkflowEngine(...)` is constructed as a plain
    attribute of the trainer in `init_workers()` (`agent_workflow_trainer.py`), and
    `initialize_pool()` builds workflow instances via a local `asyncio.Queue` +
    `ThreadPoolExecutor` — no Ray remote dispatch. `rollout_engine` (a `VerlEngine`
    client) is what makes the actual remote calls to vLLM; the engine/trainer/workflow
    objects themselves all live in the SAME TaskRunner process.

    Fix: wrap `AgentWorkflowEngine.set_training_step` (preserving its original
    episode-logging behavior) to ALSO write into `_TRAINING_STEP_STATE`, a plain
    module-level dict — safe without a lock: CPython's GIL makes a single dict-key
    write atomic enough for a monotonic counter read at task-start time, and the
    trainer's `n_parallel_tasks` workflow instances share this module's globals since
    they're threads within the SAME process (see `initialize_pool`'s
    ThreadPoolExecutor above), not separate processes. `DeepResearchWorkflow.reset()`
    reads it directly — see there."""
    from rllm.engine.agent_workflow_engine import AgentWorkflowEngine

    if getattr(AgentWorkflowEngine.set_training_step, "_dr_patched", False):
        return

    _original = AgentWorkflowEngine.set_training_step

    def _patched(self, step: int, mode: str = "train", epoch: int = 0):
        _TRAINING_STEP_STATE["step"] = int(step)
        return _original(self, step, mode=mode, epoch=epoch)

    _patched._dr_patched = True
    AgentWorkflowEngine.set_training_step = _patched


def register_periodic_push(cfg, run_dir, poll_interval_s: float = 30.0) -> None:
    """RESOLVED 2026-08-24 — call this from train_dr.py's main() BEFORE trainer.train().
    Enables a periodic 'push the latest checkpoint to HF, unscored' resilience net
    during a long real run — see hub.py::push_latest_snapshot's own docstring for the
    full reasoning (why this exists, why it's LATEST-only not best, why it's
    CPU-only/safe alongside the training GPU workload). No-op if
    cfg.push_to_hub_repo isn't set.

    FIRST ATTEMPT (superseded, kept as the record of why): hooked this into the
    AgentWorkflowEngine.set_training_step patch above, checking for new checkpoints
    on every train-mode call. REAL BUG, caught by testing rather than assumed
    correct: `set_training_step(mode="train")` fires at the START of each
    iteration, before THAT iteration's own checkpoint save happens later in the
    same iteration — so the check always ran one checkpoint behind, AND (worse)
    would have missed the run's FINAL checkpoint entirely, since there's no
    subsequent train-mode call after the last save before the run ends (only an
    automatic final val-mode call, which the `mode == "train"` gate excluded).

    FIX: a plain background thread that POLLS THE FILESYSTEM directly
    (`hub.list_checkpoint_steps`) every `poll_interval_s` seconds — sidesteps
    veRL's internal call ordering entirely. As soon as a checkpoint's files exist
    on disk, this notices within one poll interval and pushes it, regardless of
    which internal method fired it or when. Simpler AND more robust than trying to
    precisely time an internal (partly `async`) veRL hook."""
    if not getattr(cfg, "push_to_hub_repo", None):
        return

    def _poll_loop():
        import time
        import hub
        last_pushed = 0
        while True:
            time.sleep(poll_interval_s)
            try:
                steps = hub.list_checkpoint_steps(run_dir)
            except Exception as e:
                print(f"[periodic-push] couldn't list local checkpoints, will retry: {e}")
                continue
            if not steps or steps[-1] <= last_pushed:
                continue
            step = steps[-1]
            try:
                print(f"[periodic-push] merging + pushing global_step_{step} to HF "
                      f"(background, non-blocking)...")
                hub.push_latest_snapshot(cfg, run_dir, step)
                print(f"[periodic-push] done: global_step_{step} is now on HF as 'checkpoint'.")
                last_pushed = step
            except Exception as e:
                # Best-effort by design: never let a resilience feature whose entire
                # point is protecting against loss become a NEW source of failure.
                # Don't advance last_pushed on failure — retry this same step next poll.
                print(f"[periodic-push] FAILED for global_step_{step} (training continues "
                      f"unaffected — local checkpoint is still safe on disk): {e}")

    import threading
    threading.Thread(target=_poll_loop, daemon=True, name="periodic-push-poller").start()
    print(f"[periodic-push] background poller started (every {poll_interval_s:.0f}s, "
          f"pushing to {cfg.push_to_hub_repo}@checkpoint)")


# Confirmed same-process as DeepResearchWorkflow instances (see docstring above) — no
# Ray worker_process_setup_hook needed for this one, plain import-time application works.
_patch_agent_workflow_engine_step_tracking()


_REWARD_COMPONENT_ACCUMULATOR: list[dict] = []

# Which RewardInfo/env.py info_dict keys to average and log — see env.py::_terminal_reward
# for where these are computed. Deliberately excludes non-numeric/redundant keys (reward_mode
# is a string, em/f1 duplicate outcome, cite_tp/fp/fn/n_gold are components of cite_f1 itself).
_REWARD_COMPONENT_KEYS = (
    "outcome", "groundedness", "cite_f1", "cite_precision", "cite_recall",
    "cite_fabricated", "cite_uncited_claims", "hit_rate", "judge", "format_error",
    "n_steps", "base", "fab_toll", "fmt_toll", "eff_toll", "lambda_eff", "beta",
    "n_citations",
    # 2026-09-07 (RL from SFT): the behaviour axes START_HERE.md says to watch from
    # step 1 — a slide of n_steps/n_reads toward 2/0 is the outcome reward being hacked
    # by answering immediately; `answered` rising while reads hold is the intended fix
    # of the never-commit failure (F12). All set in env.py::_terminal_reward.
    "n_reads", "n_searches", "n_tool_calls", "answered", "capped",
)


def _patch_tracking_log_for_extra_metrics() -> None:
    """RESOLVED 2026-08-24 — two real gaps found while diagnosing a flat/declining
    reward trend mid-run: (1) `reward.py`'s rich per-episode breakdown (outcome,
    groundedness, cite_f1, the toll components — see env.py::_terminal_reward's
    info_dict) was computed every episode but never aggregated into anything
    visible in W&B/console, only the final scalar reward — so "is outcome
    improving while groundedness lags, or vice versa" was unanswerable from the
    logs. (2) the dead-group split (`batch/solve_none`=all-wrong,
    `batch/solve_all`=all-correct — both already logged by verl/rLLM itself,
    unrelated to our own reward.py) required manually adding two numbers by hand
    every time; no single 'what fraction of groups contributed zero GRPO signal'
    metric existed.

    Fix, both via the SAME hook: `rllm.utils.tracking.Tracking.log(self, data,
    step, ...)` is the exact call verl's trainer makes right before forwarding
    to wandb/console (`agent_workflow_trainer.py`'s `logger.log(data=metrics,
    step=self.global_steps)`) — wrapping it lets us inject additional keys into
    `data` BEFORE that forward, landing on the exact same step axis as every
    other metric verl logs. No separate wandb.log() call, no step-alignment risk.

    Reward-component averages: `DeepResearchWorkflow.run()` appends each
    episode's terminal `step_info` dict to `_REWARD_COMPONENT_ACCUMULATOR` as it
    finishes (both termination points — env-done and length-exceeded). This
    patch averages `_REWARD_COMPONENT_KEYS` across whatever accumulated since
    the last train-step log call, injects them under a `reward_components/`
    prefix, then clears the accumulator for the next step. Skipped (accumulator
    left untouched) on PURE val-mode log calls, detected POSITIVELY by checking
    for an `actor/`- or `batch/`-prefixed key (always present on any call that
    carries train-step data) rather than negatively by "any `val/`-prefixed key
    present."

    BUG (found 2026-08-24, Attempt 3 step 25, fixed same day): the original
    negative check — `is_val_call = any(k.startswith("val/") ...)` — also
    excluded the call that fires on a step where `test_freq` and
    `checkpoint_every` coincide (e.g. step 25 with both =25), where verl bundles
    THAT step's train metrics (`batch/*`, `actor/*`, ~94 keys) and the periodic
    eval's `val/*` keys into the SAME `Tracking.log()` call. Confirmed via a
    direct `wandb.Api().run(...).history()` query (not console-log guessing):
    step 25 had `has_val=True, has_reward_components=False` while every
    neighboring step had the reverse. Not permanent data loss — the accumulator
    wasn't cleared on the skipped call, so step 25's episodes silently got
    folded into step 26's average instead of reported on their own — but wrong
    on any step where `test_freq` divides evenly into `checkpoint_every`'s
    schedule. Fixed by checking positively for train-carrying keys instead.

    Dead-group roll-up: reads `batch/solve_none` + `batch/solve_all` — already
    present in `data` by the time this patch sees it, verl computes them itself
    — and injects their sum as `batch/dead_groups_pct`, the "what fraction of
    this step's groups had zero within-group reward variance (all-wrong OR
    all-correct), and therefore contributed no real GRPO advantage signal"
    number in one place instead of adding two metrics by hand each time."""
    from rllm.utils.tracking import Tracking

    if getattr(Tracking.log, "_dr_patched", False):
        return

    _original = Tracking.log

    def _patched(self, data, step, backend=None, episodes=None, trajectory_groups=None):
        # Positive check: does this call carry train-step data at all? True on
        # every normal train step AND on the mixed train+val call that fires
        # when test_freq/checkpoint_every coincide. Only false on a pure
        # val-only call, which is what we actually want to skip.
        is_train_call = any(str(k).startswith(("actor/", "batch/")) for k in data.keys())
        data = dict(data)   # don't mutate the caller's dict in place

        if is_train_call:
            if _REWARD_COMPONENT_ACCUMULATOR:
                n = len(_REWARD_COMPONENT_ACCUMULATOR)
                for key in _REWARD_COMPONENT_KEYS:
                    vals = [d[key] for d in _REWARD_COMPONENT_ACCUMULATOR if key in d
                            and isinstance(d[key], (int, float)) and not isinstance(d[key], bool)]
                    if vals:
                        data[f"reward_components/{key}"] = sum(vals) / len(vals)
                # % of rollouts that attempted >=1 citation this step — distinct from
                # the plain n_citations AVERAGE above, which conflates "half the
                # rollouts cite twice, half cite zero" with "everyone cites once."
                # This answers "how many rollouts even tried," not "how many
                # citations on average." Added 2026-08-24 per Harpreet's request
                # while diagnosing why groundedness stays at 0 across every reward
                # design tried (Probes 1-3, see TRAINING_HISTORY_LOG.md).
                cite_flags = [1.0 if d.get("n_citations", 0) > 0 else 0.0
                              for d in _REWARD_COMPONENT_ACCUMULATOR]
                if cite_flags:
                    data["reward_components/pct_rollouts_with_citation"] = (
                        sum(cite_flags) / len(cite_flags))
                data["reward_components/n_episodes"] = n
                _REWARD_COMPONENT_ACCUMULATOR.clear()

            none_ = data.get("batch/solve_none")
            all_ = data.get("batch/solve_all")
            if none_ is not None and all_ is not None:
                data["batch/dead_groups_pct"] = float(none_) + float(all_)

        return _original(self, data, step, backend=backend, episodes=episodes,
                          trajectory_groups=trajectory_groups)

    _patched._dr_patched = True
    Tracking.log = _patched


# Confirmed same-process as the trainer's own logger.log(...) calls — plain import-time
# application works, same reasoning as the two patches above.
_patch_tracking_log_for_extra_metrics()


def _ray_worker_setup_hook() -> None:
    """Ray `worker_process_setup_hook` — a plain, self-contained (no closures over
    driver-process state, so it pickles/ships cleanly) callable Ray runs once in
    EVERY new worker process at startup, before that process executes anything.
    Registered via `ray.init(runtime_env={"worker_process_setup_hook": this})` in
    `train_dr.py`'s `main()`. This is what actually gets
    `_patch_make_iterator_for_ragged_batches` into the `WorkerDict`/
    `ActorRolloutRefWorker` processes that need it — see that function's docstring
    for why merely importing this module elsewhere doesn't reach them."""
    _patch_make_iterator_for_ragged_batches()


# Also apply in THIS process (train_dr.py's own, and wherever AgentWorkflowEngine/
# DeepResearchWorkflow instances run) — cheap, idempotent, and harmless even though
# it alone doesn't reach the WorkerDict processes (the setup hook above does that).
_patch_make_iterator_for_ragged_batches()


def _task_dict_to_drtask(task: dict) -> DRTask:
    """Reconstruct a DRTask from the plain dict that survived the parquet round-trip
    (extra_info column — see module docstring). Mirrors train_dr._task_to_extra_info
    on the way out.

    NOTE: pandas/pyarrow round-trip list-typed columns as numpy arrays, not plain
    Python lists — confirmed by hitting it for real (`gold_aliases or []` raised
    "truth value of an array... is ambiguous"). Never use `or` as an
    empty-check/default on a value that might be a numpy array; use `is None` +
    explicit `list()`/`dict()` conversion instead, which works on both."""
    def _to_list(v):
        return list(v) if v is not None else []

    meta = task.get("meta")
    return DRTask(
        task_id=task["task_id"],
        question=task["question"],
        gold_answer=task["gold_answer"],
        gold_aliases=_to_list(task.get("gold_aliases")),
        supporting_titles=_to_list(task.get("supporting_titles")),
        passages=[tuple(p) for p in _to_list(task.get("passages"))],
        meta=dict(meta) if meta is not None else {},
    )


class DeepResearchAgent(BaseAgent):
    """Tracks the growing ReAct chat history and builds real Steps (carrying the
    rollout engine's actual ModelOutput, so the cumulative-prefix mask merge in
    transform.py works — see module docstring)."""

    def __init__(self):
        self._messages: list[dict] = []
        self._traj = Trajectory()

    @property
    def chat_completions(self) -> list[dict]:
        return self._messages

    @property
    def trajectory(self) -> Trajectory:
        return self._traj

    def reset(self):
        self._messages = []
        self._traj = Trajectory()

    def update_from_env(self, observation, reward, done, info, **kwargs):
        """`observation` is the tool-result text (or the opening ReAct prompt on the
        very first call) -> becomes the next 'user' turn. Empty string on the
        terminal step (env.py returns `""` once `done`), so nothing is appended
        then — correct, there's no next turn to prompt for.

        Also backfills the MOST RECENT step's observation/reward/done fields (not
        set by `Step.from_model_output`, which only knows about the model's own
        output). Purely informational/debuggability — `_process_trajectory`'s
        masking logic never reads these (only `model_output.prompt_ids`/
        `.completion_ids`), and the actual reward/termination that matters for
        training is set on `trajectory.reward` + the raised `TerminationEvent` in
        `DeepResearchWorkflow.run()`, not here. Found this gap for real (2026-08-23)
        while eyeballing rLLM's built-in episode-logger JSON output for the masking-
        gate re-verification — every step's `observation`/`done` showed empty/False
        even on a correctly-terminated episode, which was confusing to debug from
        until traced to this. Worth fixing so the NEXT person reading an episode log
        doesn't have the same false alarm."""
        if observation:
            self._messages.append({"role": "user", "content": observation})
        if self._traj.steps:
            last = self._traj.steps[-1]
            last.observation = observation
            last.reward = reward
            last.done = done

    def update_from_model(self, model_output: ModelOutput, **kwargs) -> Action:
        """Receives the FULL ModelOutput (not just `.text` — see module docstring
        for why rLLM's own MultiTurnWorkflow dropping this would silently break
        masking) and builds a real Step via the sanctioned factory."""
        text = model_output.text or model_output.content or ""
        action = Action(action=text)
        step = Step.from_model_output(model_output, messages=list(self._messages), action=action)
        _diagnose_cumulative_break(self._traj.steps, step)
        self._traj.steps.append(step)
        self._messages.append({"role": "assistant", "content": text})
        return action


def _diagnose_cumulative_break(prior_steps: list, new_step) -> None:
    """DIAGNOSTIC (added 2026-08-23, investigating the non-deterministic batch-
    divisibility bug — WORKFLOW_PORT_NOTES.md bug 11/12's real root cause is still
    open). Mirrors EXACTLY what rllm.trainer.verl.transform._process_trajectory
    checks (token-level prefix-extension of model_output.prompt_ids/completion_ids
    — NOT the coarser message-level Trajectory.is_cumulative()) so a break here is
    the actual, real cause of a trajectory splitting into >1 training row. Prints
    once per break, not fatal — this is observability, not a fix. Cheap (a list
    slice comparison), safe to leave on."""
    if not prior_steps:
        return
    prev = prior_steps[-1]
    if prev.model_output is None or new_step.model_output is None:
        return
    prev_full = list(prev.model_output.prompt_ids or []) + list(prev.model_output.completion_ids or [])
    new_prompt = list(new_step.model_output.prompt_ids or [])
    if new_prompt[: len(prev_full)] != prev_full or len(new_prompt) < len(prev_full):
        # find the first token index where they actually diverge, to see how LOCAL
        # the break is (a single-token truncation-closure mismatch vs. something
        # that desyncs much earlier)
        divergence_at = next(
            (i for i, (a, b) in enumerate(zip(prev_full, new_prompt)) if a != b),
            min(len(prev_full), len(new_prompt)),
        )
        print(
            f"[mask-diagnostic] NON-CUMULATIVE step break: prev step's full sequence "
            f"(prompt+completion) = {len(prev_full)} tokens, this step's prompt = "
            f"{len(new_prompt)} tokens, first divergence at token index "
            f"{divergence_at}/{len(prev_full)}. prev.model_output.finish_reason="
            f"{prev.model_output.finish_reason!r} (checking whether this correlates "
            f"with generation getting truncated by max_new_tokens rather than "
            f"stopping naturally — the leading hypothesis for why the chat template "
            f"re-closes the turn differently than the raw completion looked)."
        )


class DeepResearchWorkflow(TimingTrackingMixin, Workflow):
    """Drives one DeepResearchEnv episode through rLLM's Workflow contract. Mirrors
    the STRUCTURE of rLLM's own MultiTurnWorkflow (reset -> loop: generate -> env.step
    -> repeat -> terminate) but fixes the ModelOutput pass-through (see module
    docstring) — everything else (tool execution, terminal reward, credit assignment)
    is unchanged, reused straight from env.py/reward.py."""

    def __init__(self, cfg, judge=None, **kwargs):
        # rollout_engine / executor / timeout / ... are injected by AgentWorkflowEngine
        # via **kwargs -> Workflow.__init__ (confirmed: agent_workflow_engine.py calls
        # `workflow_cls(rollout_engine=..., executor=..., **workflow_args)`).
        super().__init__(**kwargs)
        self.cfg = cfg
        self.judge = judge   # None at train time — gold-based citation-F1 needs no judge (NOTES.md)
        self.agent = DeepResearchAgent()
        self.env: DeepResearchEnv | None = None   # built per-task in reset()

    def reset(self, task: dict | None = None, uid: str | None = None):
        super().reset(task, uid)   # TimingTrackingMixin: starts episode timing
        self.agent.reset()
        dr_task = _task_dict_to_drtask(task)
        self.env = DeepResearchEnv.from_dict({
            "task": dr_task,
            "cfg": self.cfg,
            "judge": self.judge,
            # RESOLVED 2026-08-24 (was always 0 — see
            # _patch_agent_workflow_engine_step_tracking's docstring for the real fix):
            # reads the live step the trainer's set_training_step() call wrote into
            # module state, not a never-set self._global_step.
            "global_step": _TRAINING_STEP_STATE["step"],
        })
        return self.env.reset()

    async def run(self, task: dict, uid: str, **kwargs) -> Episode | None:
        observation, info = await self.timed_env_call(self.reset, task=task, uid=uid)
        self.agent.update_from_env(observation, 0.0, False, info)

        max_turns = int(getattr(self.cfg, "max_turns", 8))
        # 2026-09-07: per-turn stop strings, IDENTICAL to what distill/eval_sft.py used to
        # measure the SFT numbers. Flows rllm.timed_llm_call -> VerlEngine
        # .get_token_output_from_token_input (`sampling_params.update(kwargs)`) ->
        # verl's vLLM server (`SamplingParams(max_tokens=..., **sampling_params)`), all
        # confirmed by reading the pinned sources. Without them the historical rollouts
        # let a model that free-ran past its `Action:` line have its hallucinated
        # "search results:" continuation GRADED as its own completion tokens.
        stop = list(getattr(self.cfg, "rollout_stop_sequences", ()) or ())
        llm_kwargs = dict(kwargs)
        if stop:
            llm_kwargs["stop"] = stop
        # F18 (2026-09-07): VerlEngine defaults a turn's max_tokens to
        # data.max_response_length, which is now the WHOLE-trajectory budget (4096) —
        # pass the per-turn cap explicitly or a single turn could run 4k tokens.
        llm_kwargs["max_tokens"] = int(getattr(self.cfg, "max_new_tokens", 256))
        for _ in range(max_turns + 1):   # +1: env.py's own forced-stop round
            output: ModelOutput = await self.timed_llm_call(
                self.agent.chat_completions, application_id=uid, **llm_kwargs)
            action = self.agent.update_from_model(output)

            next_obs, reward, done, step_info = await self.timed_env_call(self.env.step, action)
            self.agent.update_from_env(next_obs, reward, done, step_info)

            if output.finish_reason == "length":
                self.agent.trajectory.reward = reward
                self.agent.trajectory.info["reward_components"] = step_info
                _REWARD_COMPONENT_ACCUMULATOR.append(dict(step_info))
                raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
            if done:
                self.agent.trajectory.reward = reward
                self.agent.trajectory.info["reward_components"] = step_info
                _REWARD_COMPONENT_ACCUMULATOR.append(dict(step_info))
                raise TerminationEvent(TerminationReason.ENV_DONE)

        raise TerminationEvent(TerminationReason.MAX_TURNS_EXCEEDED)
