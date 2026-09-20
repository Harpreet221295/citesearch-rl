"""Second throughput probe (2026-08-23) — FULL cloud_preset() rollout scale
(group_size=16, prompts_per_step=1, grad_accum=4 -> 64 prompts, ~1024 raw
episodes/step), capped to 1 step. Purpose: get a REAL measured per-step time
at the actual scale the cloud run will use, after use_dynamic_bsz, instead of
extrapolating from the smaller probe_throughput.py run. Console-only logging,
no checkpoint/push side effects. Delete alongside probe_throughput.py once the
real cloud run's throughput is confirmed healthy.
"""
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import config as dr_config
import train_dr

cfg = dr_config.Config.cloud_preset()   # UNCHANGED rollout scale (group_size=16 etc.)
# PROBE_GRAD_ACCUM env var lets this script test a different rollout volume
# (e.g. grad_accum 4->2, halving tokens/step) without editing the file —
# used 2026-08-23 to test the "reduce rollout volume" lever from
# ONE_STEP_TUNING_VERL_RLLM.md §8.
if os.environ.get("PROBE_GRAD_ACCUM"):
    cfg.grad_accum = int(os.environ["PROBE_GRAD_ACCUM"])
# PROBE_LIGER=1 toggles cfg.verl_use_liger — used 2026-08-23 to measure the
# Liger fused-kernel lever from ONE_STEP_TUNING_VERL_RLLM.md §11 at full
# cloud scale, after a small-scale sanity check confirmed no crash / sane
# loss values.
if os.environ.get("PROBE_LIGER"):
    cfg.verl_use_liger = os.environ["PROBE_LIGER"] == "1"
# PROBE_TOKEN_LEN overrides cfg.verl_ppo_max_token_len_per_gpu — used
# 2026-08-24 to test additional memory headroom on top of grad_accum=2 +
# Liger, carefully (smaller increment than the original 4x jump, given
# shrinking margin) rather than another blind jump.
if os.environ.get("PROBE_TOKEN_LEN"):
    cfg.verl_ppo_max_token_len_per_gpu = int(os.environ["PROBE_TOKEN_LEN"])
# PROBE_STEPS lets this script run >1 step — used 2026-08-23 to check whether
# step 1 pays a one-time torch.compile warmup cost that steady-state (step 2+)
# doesn't, per ONE_STEP_TUNING_VERL_RLLM.md lever #1.
cfg.steps = int(os.environ.get("PROBE_STEPS", "1"))
cfg.eval_every = 999999
cfg.checkpoint_every = 999999
cfg.push_checkpoints = False
cfg.max_train_hours = 0.5          # 30min hard safety cap
cfg.log_backend = "console"
cfg.run_name = "deep_research_agent_throughput_probe_fullscale"

run_dir = Path(cfg.save_dir) / cfg.run_name
train_tasks = train_dr.build_dataset(cfg, "train")
val_tasks = train_dr.build_dataset(cfg, "eval")
train_path = train_dr.write_verl_dataset(train_tasks, run_dir / "data" / "train.parquet")
val_path = train_dr.write_verl_dataset(val_tasks, run_dir / "data" / "eval.parquet")

from dotenv import load_dotenv
load_dotenv(_HERE / ".env")

from rllm.trainer.agent_trainer import AgentTrainer
from rllm_workflow import DeepResearchWorkflow

resolved_config = train_dr.resolve_verl_config(train_dr.verl_overrides(cfg, train_path, val_path))

import ray
if not ray.is_initialized():
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env
    from rllm_workflow import _ray_worker_setup_hook
    runtime_env = get_ppo_ray_runtime_env()
    runtime_env["worker_process_setup_hook"] = _ray_worker_setup_hook
    runtime_env.setdefault("env_vars", {})
    existing_pp = os.environ.get("PYTHONPATH", "")
    runtime_env["env_vars"]["PYTHONPATH"] = str(_HERE) + (os.pathsep + existing_pp if existing_pp else "")
    ray.init(runtime_env=runtime_env, **get_ray_init_settings(resolved_config))

print(f"[probe-full] launching at {time.strftime('%H:%M:%S')} — FULL cloud scale: "
      f"rollouts/step = {cfg.group_size}x{cfg.prompts_per_step}x{cfg.grad_accum} = "
      f"{cfg.group_size*cfg.prompts_per_step*cfg.grad_accum}, steps={cfg.steps}")
t0 = time.time()
trainer = AgentTrainer(
    workflow_class=DeepResearchWorkflow,
    workflow_args={"cfg": cfg, "judge": None},
    config=resolved_config,
    train_dataset=None,
    val_dataset=None,
    backend="verl",
)
trainer.train()
print(f"[probe-full] DONE at {time.strftime('%H:%M:%S')} — total wall time for "
      f"{cfg.steps} step(s): {time.time()-t0:.1f}s")
