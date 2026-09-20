"""One-off throughput probe — NOT part of the regular CLI surface.

Purpose: measure REAL per-step wall-clock time for the actual cloud config
(Qwen2.5-3B + LoRA, real dataset, real DR env, vLLM rollout) at reduced scale,
after the use_dynamic_bsz throughput fix (2026-08-23), BEFORE committing to
the full 1000-step/6-hour cloud_preset() run again. Mirrors train_dr.py's
main() real-training path exactly (same AgentTrainer/DeepResearchWorkflow/
ray.init wiring) — just with a smaller dataset/group_size/step count and
console-only logging so it doesn't pollute the real W&B project.

Delete this file once the real cloud run's throughput is confirmed healthy —
it's a debugging tool, not a permanent fixture.
"""
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

import config as dr_config
import train_dr

cfg = dr_config.Config.cloud_preset()
# Shrink for a fast signal — same model/env/judge machinery, far less volume.
cfg.num_train_examples = 64
cfg.num_eval_examples = 8
cfg.group_size = 8
cfg.prompts_per_step = 1
cfg.grad_accum = 1
cfg.steps = 3
cfg.eval_every = 999999          # skip periodic eval — not what we're timing
cfg.checkpoint_every = 999999    # skip checkpoint saves — not what we're timing
cfg.push_checkpoints = False
cfg.max_train_hours = 0.3        # 18min hard safety cap regardless
cfg.log_backend = "console"      # no stray W&B run for a throwaway probe
cfg.run_name = "deep_research_agent_throughput_probe"

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

print(f"[probe] launching at {time.strftime('%H:%M:%S')} — "
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
print(f"[probe] DONE at {time.strftime('%H:%M:%S')} — total wall time for "
      f"{cfg.steps} steps: {time.time()-t0:.1f}s "
      f"({(time.time()-t0)/cfg.steps:.1f}s/step avg)")
