"""Config for the deep_research_agent GRPO capstone (Branch B, offline corpus).

One dataclass pins every knob so a run is reproducible from the config alone:
model + revision, LoRA, GRPO group size / prompts-per-step / grad-accum, ReAct
max_turns + search-k, the reward weights (outcome vs groundedness vs the
format/efficiency tolls), the KL coefficient, seeds, and dataset mix.

Three presets, cheapest first (COMPUTE.md tiering):
  * sanity_preset()  — tiny stand-in model, G=2, a few turns, the bundled fixture,
    MOCK judge, 1-2 GRPO steps. Proves the pipeline runs end-to-end and the mask
    wires up. DO THIS FIRST.
  * default()        — local proof-of-life on the 4060 (still small; not convergence).
  * cloud_preset()   — RunPod single A100: Qwen2.5-3B-Instruct + LoRA, G=8-16, real
    HF judge, vLLM generation via veRL. Develop local, rent to run, tear down.

The reward weights live here but the FUNCTION that combines them is Harpreet's —
reward.reward_deep_research (TODO #1). Knobs in config, policy in reward.py: tune
the shaping without editing the loss, and every run records which shaping produced it.

NOTE: the fields under "verl / rLLM wiring" are the Rosetta map to the engine — the
values map onto veRL/rLLM config keys in train_dr.py. Each is `# VERIFY` against the
installed versions (they rename keys across minor releases). See finqa_agent/verl/.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace
import json


@dataclass
class Config:
    # --- model (Instruct — the agent must already speak a chat/tool protocol) ---
    # Real target Qwen2.5-3B-Instruct (capstone brief; fits a single A100 + LoRA).
    # sanity/default swap in a small stand-in — proving the pipeline, not the number.
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = "main"
    load_in_4bit: bool = False

    # --- data (multi-hop QA; bundled per-question corpus for offline Branch B) ---
    datasets: tuple[str, ...] = ("hotpotqa", "2wikimultihopqa")
    num_train_examples: int = 128
    num_eval_examples: int = 128            # frozen held-out yardstick
    use_fixture: bool = False               # sanity_preset forces True (offline, no download)
    corpus_backend: str = "bundled"         # "bundled" (per-question ~10 paras) | "wiki_index" (cloud, full corpus)

    # --- tools / ReAct action space (search / read / answer, book §10.7) ---
    tools: tuple[str, ...] = ("search", "read", "answer")
    search_k: int = 3                       # passages returned per search
    # 2026-08-26 (distill/SFT_RL_PLAN.md §2). The opening prompt carries a full worked
    # multi-hop example — ~2/3 of its 3,136 chars. SFT's whole purpose is to move that
    # demonstration INTO THE WEIGHTS, so every post-SFT stage must run without it:
    # leaving it in makes a good result unreadable (weights learned it, or copied what
    # was in front of it?). A flag rather than an edit, so every historical run in
    # TRAINING_HISTORY_LOG.md still reproduces byte-for-byte with the default.
    # NOTE the SFT prompt and the RL/eval prompt must MATCH — see the plan doc.
    include_worked_example: bool = True
    max_turns: int = 8                      # tool-call turns before we force an answer
    max_new_tokens: int = 256               # per assistant turn
    max_obs_chars: int = 1200               # truncate a retrieved passage in the observation
                                            # (bounds the tool-token span the mask must cover)

    # --- GRPO rollout / sampling ---
    group_size: int = 8                     # G: rollouts per prompt (group-relative advantage)
    prompts_per_step: int = 2
    grad_accum: int = 4
    temperature: float = 0.9
    top_p: float = 1.0
    steps: int = 300
    adv_eps: float = 1e-6
    gen_batch_size: int = 16                # concurrent rollouts (throughput/cost lever)
    max_train_hours: float | None = None    # hard wall-clock cap (bounds cloud spend)

    # --- reward shaping (KNOBS; reward.reward_deep_research is the POLICY — TODO #1) ---
    # Layered: r = w_outcome*outcome + w_ground*groundedness
    #              - lambda_format*fmt_err - lambda_eff*(#steps).
    # Start lambda_eff at 0 and RAMP it in after the agent learns to search, so you
    # don't collapse into an answer-immediately policy that dodges the toll (the L4
    # tool-avoidance trap you hit in finqa). ramp_* control that schedule.
    reward_kind: str = "em"                 # outcome term basis: "em" (0/1) | "f1" (soft)
    # how outcome & groundedness combine (see reward.reward_deep_research):
    #   "gated"    -> outcome*(beta + (1-beta)*g)   grounding UNLOCKS outcome credit (default)
    #   "additive" -> w_outcome*outcome + w_ground*g
    reward_mode: str = "gated"
    beta: float = 0.5                       # gated: credit a right-but-ungrounded answer keeps (0..1)
    # 2026-08-24: Attempt 3 (lr=5e-5) showed a real, growing citation-avoidance exploit —
    # groundedness/cite_precision/cite_recall flatline near 0 from step 7 on, while
    # cite_fabricated ALSO drops toward 0 over the same window (not "citing badly", but
    # "learned to stop citing"). Mechanism: a static beta=0.5 lets a correct-but-uncited
    # answer bank a safe, zero-variance outcome*0.5 payoff — GRPO's group-relative
    # advantage can favor that over the noisier "try to cite, risk the fabrication toll"
    # path. Fix, mirroring lambda_eff_at's ramp pattern but the OPPOSITE direction: beta
    # RAMPS DOWN (unlike lambda_eff's ramp-up), starting early rather than late, because
    # the exploit itself showed up early (visible by step 1, entrenched by step 7) — the
    # concern lambda_eff's LATE ramp guards against (don't toll before the policy learns
    # to search) doesn't apply here; delaying the beta decay just gives the safe-uncited
    # strategy more steps to entrench before it's discouraged. See TRAINING_HISTORY_LOG.md
    # Attempt 3 for the real numbers and beta_at() below for the ramp itself.
    beta_min: float = 0.15                  # gated: floor beta decays TOWARD (shrinks the safe-uncited payoff)
    beta_ramp_start: int = 20                # step at/before which beta stays at its start value
    beta_ramp_end: int = 120                 # step at/after which beta is fully at beta_min
    # cite_gated mode (2026-08-24, Probe 4): plays beta's role for rollouts that DID
    # attempt >=1 citation (a rollout with n_citations==0 gets a hard 0 instead, see
    # reward_deep_research). Deliberately NOT ramped — see the reward.py comment at the
    # cite_gated branch for why a ramp doesn't make sense here the way it does for beta.
    cite_gated_floor: float = 0.5
    w_outcome: float = 1.0                  # additive-mode outcome weight
    w_ground: float = 0.5                   # additive-mode groundedness (citation-F1) weight
    w_fab: float = 0.25                     # EXTRA penalty per fabricated (never-retrieved) citation
    # Proof-of-Use citation grounding (citations.py measures; reward.py folds):
    w_cite: float = 0.3                     # (legacy knob; groundedness now = citation-F1 via mode above)
    require_citations: bool = True          # instruct the agent to cite (env opening prompt)
    citation_backend: str = "gold"          # "gold" (title ∈ gold supporting set — strong,
                                            # Branch-B default) | "overlap" (no-gold fallback)
                                            # | "nli" | "embedding" (stronger no-gold, pod hooks)
    citation_align_threshold: float = 0.6   # passage-claim alignment ≥ this ⇒ "supported" (measurement knob)
    lambda_format: float = 0.1
    lambda_eff: float = 0.0
    lambda_eff_max: float = 0.02
    lambda_eff_ramp_start: int = 60
    lambda_eff_ramp_end: int = 200
    numeric_tol: float = 1e-2               # (unused for QA EM; kept for the SEC-10K stretch)

    # --- judge (LLM-as-Judge groundedness) ---
    judge_backend: str = "mock"             # "mock" | "vllm" (preferred for real runs —
                                            # no HF generate(), see judge.py) | "hf" | "api"
    judge_model: str = "Qwen/Qwen2.5-3B-Instruct"

    # --- KL to reference (masked; the mask must cover BOTH PG and KL) ---
    kl_coef: float = 0.02

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    full_finetune: bool = False

    # --- optim ---
    max_len: int = 4096                     # multi-hop traces are longer than finqa's
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    bf16: bool = True
    device: str = "cuda"

    # --- reproducibility / infra ---
    seed: int = 0
    train_seed: int = 1234
    eval_seed: int = 9999                   # DISTINCT from train so held-out never overlaps
    data_dir: str = "data"
    save_dir: str = "runs"
    run_name: str = "deep_research_agent"
    log_every: int = 5
    eval_every: int = 0
    periodic_eval_n: int = 64
    checkpoint_every: int = 0
    push_checkpoints: bool = False
    resume: bool = False
    keep_best_k: int = 0
    log_backend: str = "console"
    wandb_project: str = "deep_research_agent"
    push_to_hub_repo: str | None = None
    gen_backend: str = "hf"                 # "hf" (local) | "vllm" (veRL rollout engine)

    # --- verl / rLLM wiring (Rosetta map — VERIFY keys against installed versions) ---
    verl_adv_estimator: str = "grpo"        # algorithm.adv_estimator          # VERIFY
    verl_rollout_engine: str = "vllm"       # actor_rollout_ref.rollout.name    # VERIFY
    verl_tensor_parallel: int = 1           # rollout.tensor_model_parallel_size # VERIFY
    verl_gpu_mem_util: float = 0.6          # rollout.gpu_memory_utilization    # VERIFY
    verl_max_turns: int = 8                 # multi-turn cap in the rLLM env    # VERIFY
    # 2026-08-23: fused Triton kernels (RMSNorm/RoPE/SwiGLU/cross-entropy) for
    # the actor's forward+backward — a real compute-cost reduction, not a
    # batching trick. Requires the `liger-kernel` pip package (lazy-imported
    # by verl only when this is True — verl/workers/engine/fsdp/transformer_impl.py).
    # Off by default until measured — see ONE_STEP_TUNING_VERL_RLLM.md §11.
    verl_use_liger: bool = False            # actor_rollout_ref.model.use_liger
    # 2026-08-23: dynamic_bsz token-count-per-chunk budget for the actor's
    # fwd/bwd (ref/rollout log-prob inherit it). Measured 16384->23.2GB,
    # 65536->57.8GB reserved on an 80GB A100 — see ONE_STEP_TUNING_VERL_RLLM.md
    # §6/§11. Kept as a Config field (not hardcoded) so probes/future runs can
    # tune it without editing train_dr.py.
    verl_ppo_max_token_len_per_gpu: int = 65536

    # --- 2026-09-07: RL-from-SFT wiring (see Config.rl_from_sft) ---
    # Which stage partition (splits.py) each side of training draws from. None keeps
    # the historical behaviour (data.load_tasks with num_train/eval_examples), so every
    # pre-SFT run in TRAINING_HISTORY_LOG.md still resolves byte-for-byte.
    train_split: str | None = None          # e.g. "rl_train" (14,500 q, disjoint from SFT)
    eval_split: str | None = None           # e.g. "sft_dev" (early-stop set; NEVER heldout_eval)
    # Per-turn stop strings for TRAINING rollouts. The SFT eval (distill/eval_sft.py)
    # and every diagnosis run used these so the model is cut off the moment it starts
    # writing a fake tool result; the historical GRPO rollouts had NONE, so a model that
    # free-ran past its Action: line got its hallucinated "search results:" GRADED as
    # its own tokens. Empty tuple = historical behaviour.
    rollout_stop_sequences: tuple[str, ...] = ()
    # KL-to-reference as a LOSS term (verl actor.use_kl_loss). NOTE `kl_coef` above maps
    # to algorithm.kl_ctrl.kl_coef, which verl only uses when algorithm.use_kl_in_reward
    # is True (default False) — so every historical run trained with NO KL term at all
    # (found 2026-09-07 reading the resolved config). With ref_in_actor (LoRA), the
    # reference = the loaded base weights with the adapter disabled — which is why the
    # SFT adapter is MERGED into the base first (distill/merge_sft.py): the reference is
    # then the SFT policy, not the untuned base.
    # 2026-09-07, FINDING F18 (RL_FROM_SFT_LOG.md §4.18): rLLM merges a multi-turn
    # trajectory into ONE training row whose response is [action0, obs1, action1, ...,
    # answer], then RIGHT-TRUNCATES it to verl's data.max_response_length, and only
    # writes the trajectory reward into the tensor when the UN-truncated response fits
    # (transform.py::_build_step_and_trajectory_rewards). Historically this key was set
    # to max_new_tokens=256 — a PER-TURN budget — so any episode whose merged response
    # exceeded 256 tokens was (a) trained on its first ~one turn only and (b) given
    # reward 0 regardless of its real reward. Every healthy historical run's "2-turn
    # collapse" is the policy learning the only shape that ever got paid. Split the two
    # meanings: max_new_tokens stays the per-turn generation cap (passed explicitly as
    # max_tokens each turn); this is the whole-trajectory budget. None = historical
    # behaviour (== max_new_tokens), kept so old runs resolve identically.
    verl_max_response_length: int | None = None
    verl_use_kl_loss: bool = False          # actor_rollout_ref.actor.use_kl_loss
    verl_kl_loss_coef: float = 0.001        # actor_rollout_ref.actor.kl_loss_coef (verl default)
    verl_kl_loss_type: str = "low_var_kl"   # actor_rollout_ref.actor.kl_loss_type

    # --- sanity ---
    sanity: bool = False

    def to_json(self) -> str:
        d = asdict(self)
        for k in ("tools", "lora_target_modules", "datasets", "rollout_stop_sequences"):
            d[k] = list(d[k])
        return json.dumps(d, indent=2)

    def lambda_eff_at(self, step: int) -> float:
        """Efficiency toll at a given step — linearly ramped 0 -> lambda_eff_max
        between ramp_start and ramp_end, so the toll doesn't punish searching before
        the policy has learned to search (the L4 tool-avoidance trap)."""
        if step <= self.lambda_eff_ramp_start:
            return self.lambda_eff
        if step >= self.lambda_eff_ramp_end:
            return self.lambda_eff_max
        frac = (step - self.lambda_eff_ramp_start) / max(
            1, self.lambda_eff_ramp_end - self.lambda_eff_ramp_start)
        return self.lambda_eff + frac * (self.lambda_eff_max - self.lambda_eff)

    def beta_at(self, step: int) -> float:
        """Gated-reward beta at a given step — linearly ramped DOWN from `beta` to
        `beta_min` between beta_ramp_start and beta_ramp_end. Opposite direction from
        lambda_eff_at (which ramps UP): beta starts at its most-forgiving value (credits
        a correct-but-uncited answer generously, so early training isn't punished for a
        citation skill it hasn't learned yet) and shrinks that floor as training
        progresses, so staying uncited stops being a viable safe strategy. See the
        `beta_min`/`beta_ramp_*` fields above for the real evidence motivating this."""
        if step <= self.beta_ramp_start:
            return self.beta
        if step >= self.beta_ramp_end:
            return self.beta_min
        frac = (step - self.beta_ramp_start) / max(
            1, self.beta_ramp_end - self.beta_ramp_start)
        return self.beta + frac * (self.beta_min - self.beta)

    @staticmethod
    def sanity_preset() -> "Config":
        """Overfit the bundled fixture for 1-2 GRPO steps, mock judge. Reward should
        MOVE, nothing NaNs, and the mask check must pass. Proves the wiring BEFORE
        any real run."""
        return Config(
            sanity=True,
            use_fixture=True,
            corpus_backend="bundled",
            judge_backend="mock",
            num_train_examples=4,
            num_eval_examples=4,
            group_size=2,
            prompts_per_step=2,
            grad_accum=1,
            max_turns=4,
            search_k=3,
            max_new_tokens=160,
            steps=2,
            run_name="deep_research_agent_sanity",
        )

    @staticmethod
    def default() -> "Config":
        """Local proof-of-life on the 4060 (small stand-in; NOT convergence)."""
        return Config()

    @staticmethod
    def cloud_preset() -> "Config":
        """RunPod single A100: the real Qwen2.5-3B + LoRA run, vLLM judge, vLLM rollout."""
        return Config(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            use_fixture=False,
            corpus_backend="bundled",       # start bundled (comparable loop); flip to
                                            # "wiki_index" for the paper-comparable number
            judge_backend="vllm",           # batched vLLM, not HF generate() — see judge.py
            judge_model="Qwen/Qwen2.5-3B-Instruct",
            num_train_examples=2048,
            num_eval_examples=256,
            group_size=16,
            prompts_per_step=1,
            # RESOLVED 2026-08-23/24: real-cloud-scale throughput tuning (see
            # ONE_STEP_TUNING_VERL_RLLM.md) measured 6min/step at grad_accum=4 —
            # 1000 steps would've taken ~100h against a 6h budget. grad_accum=2
            # (half rollout volume, 512 vs 1024 raw trajectories/step) is the
            # ONE lever in that tuning pass that trades GRPO batch-statistics
            # quality for wall-clock speed (every other fix was pure waste
            # removal) — kept explicit here rather than silently baked in.
            # Measured result at grad_accum=2 + use_liger=True (below) + the
            # verl_* overrides in verl_overrides(): 85.7s/step steady-state,
            # ~252 steps reachable in the 6h cap (clears lambda_eff_ramp_end=200).
            grad_accum=2,
            gen_batch_size=16,
            steps=1000,
            max_turns=8,
            max_new_tokens=256,
            gen_backend="vllm",             # veRL drives vLLM rollouts on the pod
            bf16=True,
            max_train_hours=6.0,
            # RESOLVED 2026-08-24: TWO real attempts killed over this lr search, both with
            # direct evidence, not guessing:
            #   1st attempt, lr=1e-5 (the base Config default): killed at step 31/252 —
            #     reward growth flattening early (10-step-window avg 0.048->0.092->0.099,
            #     decile 2->3 gain much smaller than 1->2) alongside a rising dead-group
            #     fraction (28%->59% none+all groups). Matched finqa_agent's own
            #     documented lr=1e-5 failure (last_runpod_session.md: flat reward, FAILED
            #     its eval gate at 88 steps) — that session's own A/B probe found lr=1e-4
            #     gave real movement where lr=1e-5 was flat, so tried that next.
            #   2nd attempt, lr=1e-4: killed at step 28/252 — WORSE, a real collapse, not
            #     just slow: batch/solve_none hit 96-100% for 4 straight steps,
            #     response_length/clip_ratio 96-98% (stopped terminating cleanly, hitting
            #     max_new_tokens every rollout), val/pass@1 dropped to 0.0 at the step-25
            #     eval (vs 0.25 for the lr=1e-5 run at the same step). Too aggressive a
            #     jump for this model/task/single-epoch-GRPO combination — grad_norm/KL
            #     looked stable throughout (no obvious blow-up signal there), but the
            #     actual generation behavior degenerated regardless; a reminder that
            #     "no gradient spike" isn't the same as "training is healthy."
            # lr=5e-5 (halfway between the two, in log-space close to the failed 1e-4)
            # is the next real data point, not a final answer — re-litigate if this ALSO
            # shows the same collapse pattern (see rllm_workflow.py's reward_components/*
            # + batch/dead_groups_pct + response_length/clip_ratio, all logged to W&B
            # now specifically because of this search).
            lr=5e-5,
            log_backend="wandb",
            eval_every=25,
            periodic_eval_n=128,
            keep_best_k=2,
            checkpoint_every=25,
            push_checkpoints=True,
            push_to_hub_repo="harpreet22happy/deep-research-agent-grpo",
            run_name="deep_research_agent_cloud",
            # RESOLVED 2026-08-24: fused Triton kernels for the actor's fwd/bwd —
            # free, no training-signal tradeoff (unlike grad_accum above).
            # Measured: update_actor -13.3%, total step -3.3% (see
            # ONE_STEP_TUNING_VERL_RLLM.md §9). Sanity-checked (no crash, sane
            # loss/grad_norm) before this full-scale measurement.
            verl_use_liger=True,
            # verl_ppo_max_token_len_per_gpu left at its Config default (65536,
            # not the 98304 also tested) — that jump measured only -3% further
            # speedup for +21% peak memory (57GB->69GB, margin 23GB->11GB) —
            # a bad risk/reward trade, not applied.
            #
            # RESOLVED 2026-08-24: lambda_eff wiring bug fixed (was permanently 0
            # regardless of step — see rllm_workflow.py's
            # _patch_agent_workflow_engine_step_tracking, verified end-to-end with
            # a real run: global_step correctly threaded 1->2->3->4 into
            # DeepResearchEnv.from_dict). Since the ramp now actually engages,
            # revisited WHEN it should — Harpreet's own prior-lab experience:
            # penalizing turn-count before the policy has learned to use tools at
            # all can suppress tool-calling behavior early, not just make it more
            # efficient. Pushed the ramp back from the base default (60->200) to
            # start only after the model has had real runway to learn the task
            # first. At ~85.7s/step steady-state, ~252 steps fit the 6h budget —
            # ramp_end=230 leaves ~22 steps at full toll strength rather than
            # cutting the ramp off mid-climb.
            lambda_eff_ramp_start=150,
            lambda_eff_ramp_end=230,
            # ADDED 2026-08-24, Attempt 3 (lr=5e-5) real finding: citation avoidance —
            # groundedness/cite_precision/cite_recall flatline near 0 from step 7 on,
            # while cite_fabricated ALSO drops toward 0 over the same window (the model
            # learned to stop citing, not to cite badly). See TRAINING_HISTORY_LOG.md
            # Attempt 3 for the full numbers and Config.beta_at's docstring for the
            # mechanism. Ramp starts EARLY (not late like lambda_eff) because the exploit
            # itself showed up early — delaying the decay just gives the safe-uncited
            # strategy more steps to entrench. beta_min left at the Config default (0.15)
            # for now — a first real data point, not a tuned final value.
            beta_ramp_start=20,
            beta_ramp_end=120,
        )

    # --- 2026-08-24: short (25-step) reward-design comparison probes ---
    # Not real training runs — quick, cheap, apples-to-apples diagnostics to see how
    # different reward shapings actually behave before committing 100+ steps to one.
    # Each starts FRESH from the base model (resume stays False, distinct run_name so
    # checkpoint dirs/W&B runs don't collide or chain off each other), same lr=5e-5 as
    # the only non-collapsing value found so far — isolate the reward-design variable,
    # don't reintroduce the lr variable. No periodic eval/checkpoint push (steps=25 is
    # too short for either to matter) — read reward_components/*, batch/dead_groups_pct,
    # response_length/clip_ratio from training itself, same signals that already caught
    # every real finding this session. See TRAINING_HISTORY_LOG.md for the results.

    @staticmethod
    def probe_beta0() -> "Config":
        """Candidate 1 — beta=0.0 static (no ramp; beta_min=0.0 too so beta_at() is a
        no-op constant). The most aggressive anti-citation-avoidance setting: a
        correct-but-uncited answer earns ZERO credit, not the 50% floor Attempts 1-4
        used. Real risk to watch: this could over-suppress outcome-learning entirely
        rather than just fixing citation — watch outcome/hit_rate, not just groundedness."""
        return replace(
            Config.cloud_preset(),
            run_name="deep_research_agent_probe_beta0",
            steps=25,
            beta=0.0, beta_min=0.0,
            eval_every=0, checkpoint_every=0, push_checkpoints=False,
        )

    @staticmethod
    def probe_beta_ramp_fast() -> "Config":
        """Candidate 2 — beta ramped 0.5->0.0 over steps 1->30 (vs. cloud_preset's
        slower 20->120 ramp to 0.15). Steeper, starts immediately, reaches a full 0.0
        floor — a real test of 'ramp fast and all the way down' as its own candidate,
        not a tweak of Attempt 4's schedule."""
        return replace(
            Config.cloud_preset(),
            run_name="deep_research_agent_probe_beta_ramp_fast",
            steps=25,
            beta=0.5, beta_min=0.0, beta_ramp_start=0, beta_ramp_end=30,
            eval_every=0, checkpoint_every=0, push_checkpoints=False,
        )

    @staticmethod
    def probe_additive() -> "Config":
        """Candidate 3 — reward_mode="additive" (r = w_outcome*outcome +
        w_ground*groundedness - tolls) instead of the gated formula. No safe-uncited
        floor to exploit at all — outcome and groundedness are independently summed,
        not one gating the other. w_outcome/w_ground left at Config defaults (1.0/0.5)."""
        return replace(
            Config.cloud_preset(),
            run_name="deep_research_agent_probe_additive",
            steps=25,
            reward_mode="additive",
            eval_every=0, checkpoint_every=0, push_checkpoints=False,
        )

    @staticmethod
    def probe_cite_gated() -> "Config":
        """Candidate 4 — reward_mode="cite_gated": a HARD ZERO for any rollout with
        n_citations==0 (not the soft beta floor Probes 1-3 all showed is exploitable —
        beta gives the SAME partial credit to a non-attempt as to a bad attempt, making
        abstention as safe as trying and failing). Rollouts that DID cite something
        still get the same quality-scaled credit as "gated" mode
        (cite_gated_floor + (1-cite_gated_floor)*groundedness) — this closes the
        specific abstention loophole without removing the pressure toward CORRECT
        citation the way a pure attempt-only gate would (Harpreet's catch: a naive
        "any citation, credit for trying" rule would only teach the model to paste
        garbage citations, not correct ones). No ramp — see reward.py's cite_gated
        branch for why."""
        return replace(
            Config.cloud_preset(),
            run_name="deep_research_agent_probe_cite_gated",
            steps=25,
            reward_mode="cite_gated",
            eval_every=0, checkpoint_every=0, push_checkpoints=False,
        )

    # --- 2026-09-07: GRPO from the SFT checkpoint (the stage START_HERE.md calls next) ---
    @staticmethod
    def rl_from_sft() -> "Config":
        """GRPO starting from the SFT policy, on questions SFT never saw.

        HOW THE SFT POLICY GETS IN: `model_name` points at a LOCAL directory holding the
        base model with the SFT LoRA MERGED into its weights (distill/merge_sft.py), and
        GRPO trains a FRESH LoRA (B=0 init, so step 0 == the SFT policy exactly) on top.
        Not verl's `lora_adapter_path` — that loads the adapter into the actor but makes
        the KL reference the adapter-DISABLED base, i.e. the untuned model, so the KL term
        would actively pull the policy AWAY from what SFT installed.

        THE TARGET (START_HERE.md F12): 16% of SFT episodes never commit to an answer.
        THE RISK (same doc): the cheapest way to stop collecting zeros is to answer
        immediately and stop reading — Probe 3's collapse shape. Three counters here:
          * reward_mode="cite_gated" — a rollout with no citation gets a hard 0, and a
            citation only SCORES if that passage was `read` (citations.py), so the read
            step stays load-bearing for reward.
          * KL loss to the SFT reference (use_kl_loss) — the previous runs had no KL term
            at all (see the verl_use_kl_loss field note).
          * turns / reads / searches / answered logged per step under reward_components/
            so a drift toward 2-turn episodes is visible inside the first ~20 steps.
        Prompt: include_worked_example=False — MUST match what the adapter was trained
        under (SFT_RL_PLAN.md §2); silently wrong otherwise."""
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        return replace(
            Config.cloud_preset(),
            run_name="deep_research_agent_rl_from_sft_correct_only",
            model_name=os.path.join(here, "sft_merged"),
            include_worked_example=False,
            train_split="rl_train",
            eval_split="sft_dev",
            num_train_examples=14_500,       # informational — splits.py fixes the size
            num_eval_examples=128,           # sft_dev[:128], the periodic-eval slice
            periodic_eval_n=128,
            rollout_stop_sequences=("\nThought:", "\nsearch results:", "\n["),
            reward_mode="cite_gated",
            verl_use_kl_loss=True,
            verl_kl_loss_coef=0.01,          # 10x verl's default; first data point, watch actor/kl_loss
            verl_kl_loss_type="low_var_kl",
            # The SFT policy does ~5.4 tool calls/episode (RESULTS.md §1), so the growing
            # conversation is far longer than the 2-turn episodes 4096 was sized for.
            max_len=6144,
            # F18: the merged multi-turn response. SFT trajectories average ~1,300
            # response tokens (1,705 incl. prompt, max 3,381 in validate.py's scan);
            # 4096 covers the 8-turn worst case with room. vLLM max_model_len becomes
            # max_len + this = 10,240.
            verl_max_response_length=4096,
            # Run length. NOTE (issue #15, 2026-09-07): max_train_hours is NOT enforced
            # anywhere in train_dr.py/rllm_workflow.py — it is a config field only; every
            # historical cloud run was killed by hand. So the budget comes from `steps`:
            # measured 184-191 s/step at 512 trajectories/step (probe, W&B mchhqpc6) ->
            # 100 steps ~= 5.3 h, checkpoints + sft_dev eval at 25/50/75/100.
            steps=100,
            lr=5e-5,                          # the only lr that neither flat-lined nor collapsed (Attempts 1-3)
            temperature=0.9,
        )

    @staticmethod
    def probe_rl_from_sft() -> "Config":
        """Short wiring + behaviour probe of rl_from_sft: does the merged model load in
        vLLM and FSDP, do stop sequences apply, is actor/kl_loss logged, how long is a
        step at the SFT policy's ~5 turns/episode, and — the thing to actually read —
        do reward_components/n_steps and n_reads hold steady or slide toward 2-turn
        episodes. No checkpoints, no push, no periodic eval."""
        return replace(
            Config.rl_from_sft(),
            run_name="deep_research_agent_probe_rl_from_sft_correct_only",
            steps=12,
            # checkpoint once at step 10 (no push): exercises the verl checkpoint ->
            # hub.merge_checkpoint -> distill/eval_rl.py chain on a real RL checkpoint
            # BEFORE the 6h run depends on it.
            eval_every=0, checkpoint_every=10, push_checkpoints=False,
            max_train_hours=1.0,
        )
