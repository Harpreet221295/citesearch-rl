"""hub.py — checkpoint merge (via verl's OWN `model_merger`) + HF Hub push, with a
proper per-run branch name (+ a DD_MM_YYYY__HH_MM_SS timestamp) so reruns don't
collide, mirroring RUNPOD_PLAYBOOK.md pattern #5b/#6 (keep BEST and LATEST separately,
mirror to HF so a checkpoint survives a full pod wipe).

Why a separate module: veRL's `AgentTrainer` owns the actual training loop (dispatched
to Ray, opaque to us) and writes its OWN sharded checkpoints under
`trainer.default_local_dir/global_step_N/actor` — there's no simple per-step Python
callback to hook a push into (unlike finqa's hand-rolled loop, which called
`save_checkpoint`/`maybe_save_best` explicitly every iteration). So here, checkpoint
handling is a SEPARATE, explicit step:
    merge (verl.model_merger — LoRA-aware, correctly reconstructs a standard PEFT
           adapter from the FSDP shards; reusing the framework's own tool instead of
           hand-rolling FSDP-shard reconstruction, which is genuinely easy to get
           wrong) -> push (our own branch-per-run + best/latest layout).
`verl.model_merger` DOES support `--hf_upload_path` directly, but only pushes to a
repo's default branch (no `revision`/branch control) — so we merge locally with it,
then push ourselves via `huggingface_hub` for branch + timestamp control.

Orchestration (which checkpoints to evaluate + keep) lives in train_dr.py, which
already needs evaluate.py's batched-vLLM scorer; this module is just the mechanics
of "get one local checkpoint into HF Hub."
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


def run_timestamp() -> str:
    """DD_MM_YYYY__HH_MM_SS — used to make each run's HF branch unique across reruns
    (a plain `run_name` branch, like finqa uses, would collide if you rerun with the
    same run_name; a fresh timestamp per run can't)."""
    return datetime.now().strftime("%d_%m_%Y__%H_%M_%S")


def run_branch_name(cfg, run_dir: Path) -> str:
    """The HF branch for THIS run: `{run_name}__{timestamp}`, captured ONCE into
    `run_dir/run_meta.json` so repeated push calls during/after the same run (e.g.
    a periodic push during training, then a final one after) land on the SAME
    branch rather than minting a new one each time. `--resume` also reads this so a
    resumed run keeps pushing to its original branch."""
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if "run_branch" not in meta:
        meta["run_branch"] = f"{cfg.run_name}__{run_timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))
    return meta["run_branch"]


def list_checkpoint_steps(local_dir: Path) -> list[int]:
    """veRL writes `global_step_<N>/actor` (LoRA/GRPO — no critic dir) under
    `local_dir` (= `trainer.default_local_dir`, our `save_dir/run_name`). Returns
    the sorted step numbers that have a real `actor` checkpoint."""
    if not local_dir.exists():
        return []
    steps = []
    for p in local_dir.iterdir():
        m = re.match(r"global_step_(\d+)$", p.name)
        if m and (p / "actor").exists():
            steps.append(int(m.group(1)))
    return sorted(steps)


def merge_checkpoint(step_dir: Path, target_dir: Path, backend: str = "fsdp") -> Path:
    """Reconstruct a standard HF-format LoRA adapter from one veRL sharded checkpoint
    step, via veRL's OWN `python -m verl.model_merger` (LoRA-aware: reads
    `lora_train_meta.json` + the state dict's `lora_*` params, writes a proper
    `adapter_config.json` + `adapter_model.safetensors` under `target_dir/lora_adapter/`
    — a standard PEFT adapter dir, loadable via `PeftModel.from_pretrained` or vLLM's
    `LoRARequest`). Returns that adapter dir.
    `step_dir` = e.g. `runs/<run_name>/global_step_200` (contains an `actor/` subdir).

    RESOLVED 2026-08-24 — a real bug hit while testing the periodic-push feature, not
    hypothetical: `verl.model_merger merge`'s CLI ALWAYS tries to also save a full
    dense HF model after successfully extracting the LoRA adapter
    (`base_model_merger.py::save_hf_model_and_tokenizer` calls `save_lora_adapter()`
    — which correctly writes the adapter, `.pop()`ing the `lora_*` keys out of
    `state_dict` as it goes — then UNCONDITIONALLY calls `model.save_pretrained(...,
    state_dict=state_dict)` on whatever's left). Since `checkpoint.save_lora_only=True`
    (see train_dr.py) means the checkpoint never HAD base-model weights to begin with,
    `state_dict` is empty after the LoRA keys are popped, the full-model save
    legitimately produces nothing, and the tool's own `validate_hf_model_output` check
    raises — even though the adapter itself was already written successfully before
    that point. Confirmed directly: `lora_adapter/adapter_model.safetensors` (504 real
    LoRA tensors, correct r=16/alpha=32 in `adapter_config.json`) exists on disk despite
    the subprocess's non-zero exit. Fix: don't trust the exit code — check for the
    actual adapter output we need instead, which is unaffected by the full-model-save
    failure since it happens earlier in the same process."""
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "verl.model_merger", "merge",
        "--backend", backend,
        "--local_dir", str(step_dir / "actor"),
        "--target_dir", str(target_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    adapter_dir = target_dir / "lora_adapter"
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        # A real failure (not the known "full-model save on an adapter-only
        # checkpoint" case above, since that one still produces the adapter file
        # before failing) — surface the subprocess's own output for diagnosis.
        raise RuntimeError(
            f"verl.model_merger did not produce {adapter_file} — either this wasn't a "
            f"LoRA run (full_finetune=True: push target_dir itself, not lora_adapter), "
            f"the merger's output layout changed in a newer verl, or a genuine merge "
            f"failure. subprocess exit={result.returncode}\nstdout:\n{result.stdout[-2000:]}"
            f"\nstderr:\n{result.stderr[-2000:]}")
    if result.returncode != 0:
        print(f"[hub] verl.model_merger exited non-zero ({result.returncode}) but "
              f"{adapter_file} was written successfully — the known LoRA-only-checkpoint "
              f"quirk (see merge_checkpoint's docstring), not a real failure. Proceeding.")
    return adapter_dir


def push_folder_to_hub(local_dir: Path, repo_id: str, branch: str, path_in_repo: str,
                        private: bool = True) -> None:
    """Push `local_dir`'s contents to `repo_id` at `revision=branch`, under
    `path_in_repo` (e.g. "checkpoint" for LATEST, "best/step200_em0.340" for a BEST
    snapshot — RUNPOD_PLAYBOOK.md pattern #5b: two separate artifacts, don't conflate
    them). Creates the repo/branch if they don't exist yet (idempotent — safe to call
    repeatedly as training progresses)."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, revision=branch,
                       path_in_repo=path_in_repo)
    print(f"[hub] pushed {local_dir} -> {repo_id}@{branch}:{path_in_repo}")


def push_latest_snapshot(cfg, run_dir: Path, step: int, private: bool = True) -> dict:
    """RESOLVED 2026-08-24 — periodic mid-training resilience push (was: HF push only
    happened ONCE, after the whole run finished — a real gap against
    RUNPOD_PLAYBOOK.md's own "survive a full pod wipe" checkpoint-mirroring pattern; a
    genuinely catastrophic loss DURING a long run would have had zero HF backup yet).

    Merges ONE local veRL checkpoint step to a standard adapter (`merge_checkpoint` —
    CPU-only, verified via `verl/model_merger/fsdp_model_merger.py` having no CUDA
    calls at all, so this is safe to run alongside the actively-training GPU workload)
    and pushes it to the run's `checkpoint/` path (LATEST only — deliberately NOT
    scored/ranked against `keep_best_k` here; that needs a real eval pass via a
    SEPARATE vLLM engine, which would compete for GPU memory with the training run's
    own resident engine. `push_checkpoints()` still does that properly, once, after
    `trainer.train()` completes and the training-time engine is freed — this function
    is purely "don't lose everything if the run dies before then").

    Writes a small `checkpoint_info.json` alongside the adapter — the resume-relevant
    context a future session pulling this down needs, since the pushed adapter alone
    (a standard PEFT dir) carries no metadata of its own: which step it's from, when
    it was pushed, the run's HF branch (so a resumed run keeps using the SAME branch,
    not a fresh one), and the key hyperparameters (lambda_eff ramp, grad_accum, model)
    needed to reconstruct a matching training config rather than guess.

    NOTE on what this does/doesn't enable: this is a WARM-START snapshot (fresh LoRA
    weights to continue training FROM), not a byte-exact veRL resume (which needs the
    full local sharded checkpoint — optimizer state, RNG state — under
    `run_dir/global_step_N/`, which already lives on `/workspace` and survives a plain
    pod STOP on its own; veRL's own `resume_mode=auto` is the primary resume path for
    that common case). This function's real purpose is the harder case: local disk
    itself is lost (not just a stop) — then a warm-start from the last pushed adapter
    beats losing all training progress outright, even without full optimizer-state
    continuity."""
    if not cfg.push_to_hub_repo:
        return {}
    branch = run_branch_name(cfg, run_dir)
    merged_dir = run_dir / "_merged" / f"step{step}_latest"
    adapter_dir = merge_checkpoint(run_dir / f"global_step_{step}", merged_dir)
    info = {
        "step": step,
        "pushed_at": datetime.now().isoformat(),
        "run_branch": branch,
        "run_name": cfg.run_name,
        "model_name": cfg.model_name,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "grad_accum": cfg.grad_accum,
        "group_size": cfg.group_size,
        "lambda_eff_ramp_start": cfg.lambda_eff_ramp_start,
        "lambda_eff_ramp_end": cfg.lambda_eff_ramp_end,
        "note": "warm-start snapshot (latest, unscored) — see hub.py::push_latest_snapshot",
    }
    (adapter_dir / "checkpoint_info.json").write_text(json.dumps(info, indent=2))
    push_folder_to_hub(adapter_dir, cfg.push_to_hub_repo, branch, "checkpoint", private=private)
    return info


def pull_folder_from_hub(repo_id: str, branch: str, path_in_repo: str, local_dir: Path) -> Path:
    """The --resume-after-pod-wipe counterpart (RUNPOD_PLAYBOOK.md pattern #6): pull
    a previously-pushed checkpoint/adapter dir back down when local disk is empty."""
    from huggingface_hub import snapshot_download
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, revision=branch, allow_patterns=f"{path_in_repo}/*",
                       local_dir=str(local_dir))
    return local_dir / path_in_repo
