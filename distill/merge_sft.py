"""Merge the SFT LoRA adapter into the base weights -> a standalone HF model directory.

    python distill/merge_sft.py                       # pull the Hub adapter, merge, verify
    python distill/merge_sft.py --adapter distill/runs/sft_full/final   # a local adapter

WHY MERGE AT ALL (2026-09-07). verl 0.9.0 can load an existing adapter into the actor
(`actor_rollout_ref.model.lora_adapter_path`), but with LoRA it computes the reference
policy as "the actor with the adapter DISABLED" (`ref_in_actor`, ray_trainer.py:356) —
i.e. the untuned base. A KL term against that reference pulls the policy back toward
the model that never reads before citing, which is the exact behaviour SFT installed.
Merging the adapter into the weights and training a FRESH LoRA on top makes the
reference the SFT policy itself, and the fresh LoRA's B=0 init means step 0 of GRPO is
byte-for-byte the SFT policy.

WHAT IS VERIFIED, not assumed: the merged model's logits are compared against the
un-merged (base + adapter) model's on a real student prompt — max |Δlogit| and argmax
agreement are printed. bf16 rounding makes a small Δ normal; a large one, or argmax
disagreement, means the merge is wrong and nothing downstream should run.

The output directory is gitignored (~6 GB); it is rebuildable from the base model +
`harpreet22happy/deep-research-agent-sft` in about a minute.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE), str(_HERE / "distill")):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as dr_config

HUB_ADAPTER = "harpreet22happy/deep-research-agent-sft"
DEFAULT_OUT = _HERE / "sft_merged"
DEFAULT_ADAPTER_DIR = _HERE / "sft_adapter"


def download_adapter(repo: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(local_dir))
    assert (local_dir / "adapter_config.json").exists(), f"no adapter_config.json in {local_dir}"
    return local_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help=f"local adapter dir; default: download {HUB_ADAPTER} to {DEFAULT_ADAPTER_DIR}")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cfg = dr_config.Config.cloud_preset()
    out = Path(args.out)
    if out.exists() and (out / "config.json").exists() and not args.force:
        raise SystemExit(f"{out} already exists — pass --force to rebuild it.")

    adapter = Path(args.adapter) if args.adapter else download_adapter(HUB_ADAPTER, DEFAULT_ADAPTER_DIR)
    acfg = json.loads((adapter / "adapter_config.json").read_text())
    print(f"base    : {cfg.model_name}")
    print(f"adapter : {adapter}  (r={acfg.get('r')}, alpha={acfg.get('lora_alpha')}, "
          f"targets={sorted(acfg.get('target_modules', []))})")

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    # Reference = bf16 base + bf16 adapter, UN-merged — what vLLM served during eval_sft.py.
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.bfloat16,
                                                device_map={"": 0})
    peft_model = PeftModel.from_pretrained(base, str(adapter))
    peft_model.eval()

    # --- reference logits from the UN-merged model, on a real student prompt ---
    import env as env_mod
    from types import SimpleNamespace
    lean = replace(cfg, include_worked_example=False)
    q = "What nationality is the director of the film Blue Harvest?"
    prompt = env_mod._opening_prompt(SimpleNamespace(question=q), lean)
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": "Thought: I need the director first.\n"
                                             "Action: search[who directed Blue Harvest]"}]
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt")
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = ids.to("cuda")
    with torch.no_grad():
        ref = peft_model(input_ids=ids).logits.float().cpu()

    # --- merge in fp32, cast ONCE at the end ---
    # 2026-09-07: the first merge was done in bf16 (peft computes B@A*scale and adds it
    # to the bf16 weight, both at ~3 significant digits) and verified at max |dlogit|
    # 2.125 / mean 0.066 / argmax agreement 0.991 against the un-merged model. Merging in
    # fp32 rounds once instead of three times.
    del peft_model, base
    torch.cuda.empty_cache()
    base32 = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.float32,
                                                  device_map={"": 0})
    merged = PeftModel.from_pretrained(base32, str(adapter)).merge_and_unload()
    merged = merged.to(torch.bfloat16)
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    # generation_config comes with save_pretrained; make sure the chat template did too
    assert (out / "config.json").exists()
    print(f"merged  -> {out}  ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file())/2**30:.2f} GiB)")

    # --- verify: reload from disk (what vLLM/FSDP will see) and compare ---
    del merged, base32
    torch.cuda.empty_cache()
    re = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16, device_map={"": 0})
    with torch.no_grad():
        got = re(input_ids=ids).logits.float().cpu()
    d = (got - ref).abs()
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"\nVERIFY merged-from-disk vs base+adapter on {ids.shape[1]} positions:")
    print(f"  max |dlogit|  = {d.max().item():.4f}   mean |dlogit| = {d.mean().item():.5f}")
    print(f"  argmax agreement = {agree:.4f}")
    # Thresholds: argmax agreement is the one that matters for greedy behaviour; the max
    # logit delta is a tail statistic over ~50M logits and bf16 alone moves it by ~2.
    # The DECIDING check is behavioural: distill/eval_rl.py with no adapter on sft_dev
    # must reproduce the adapter's eval_sft.py numbers.
    ok = agree >= 0.98 and d.max().item() < 4.0
    print(f"  => {'PASS' if ok else 'FAIL — do not train on this merge'}  "
          f"(then confirm behaviourally: eval_rl.py --split sft_dev --n 128, no adapter)")
    (out / "MERGE_INFO.json").write_text(json.dumps({
        "base": cfg.model_name, "adapter": str(adapter), "hub_adapter": HUB_ADAPTER,
        "adapter_config": acfg, "verify": {"max_abs_dlogit": d.max().item(),
                                           "mean_abs_dlogit": d.mean().item(),
                                           "argmax_agreement": agree, "pass": ok}}, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
