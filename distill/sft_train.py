"""LoRA SFT for the deep-research student. peft + a plain torch loop.

    python distill/sft_train.py --tune-batch      # find the largest batch that fits, then exit
    python distill/sft_train.py --overfit 16      # sanity: drive loss to ~0 on 16 examples
    python distill/sft_train.py                   # the real run

WHY NOT TRL / HF Trainer / unsloth (decided 2026-08-26):
  * The masking is the highest-risk component and it is already built and verified
    (18 tests in distill/tests). TRL's SFTTrainer and HF Trainer both want to own
    tokenization and collation; handing them our tensors invites them to re-derive the
    boundaries their own way. That failure is silent — loss falls, model learns the wrong
    thing — which is the exact class of bug this session has spent its time on.
  * `assignments/sft_min` is this repo's reference SFT pattern and is a plain torch loop
    with its own build_labels / sft_loss / assert_masking_correct. Same shape here.
  * Early stopping has to be on GENERATION metrics (does it call read? does it cite what
    it read?), which means multi-turn agent rollouts. Trainer's eval loop computes a loss
    over a dataset; it cannot do that, so the loop buys little.
  * unsloth: dry-run showed it leaves torch/vllm/verl/flash-attn alone but downgrades
    datasets 5.0.1 -> 4.3.0 and transformers 5.5.4 -> 5.5.0 and adds xformers. The dataset
    is ~1.4k examples x ~900 tokens — minutes per epoch either way — so it would trade a
    re-verification of the RL stack for ~15 minutes. Revisit if we ever get compute-bound.

BATCH SIZE (--tune-batch): measured on the actual A100, not guessed. It walks powers of
two, runs a real forward+backward at the dataset's p99 sequence length (the worst case
that will actually occur), and reports peak memory. It deliberately probes with the
LONGEST sequences rather than the mean — a batch tuned on 900-token averages will OOM
partway through an epoch when a 2,200-token example lands, which is the annoying kind of
failure because it happens late.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE), str(_HERE / "distill")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.utils.data import DataLoader, Dataset

import config as dr_config
from sft_data import IGNORE_INDEX

DATASET = _HERE / "distill" / "sft_dataset.pt"
OUTDIR = _HERE / "distill" / "runs"


class TrajDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids, self.labels = input_ids, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def collate(batch, pad_id: int):
    """Right-pad. Padding is IGNORE_INDEX in labels so it contributes no loss, and 0 in
    the attention mask so it is not attended to — both, not either."""
    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        labs.append(b["labels"] + [IGNORE_INDEX] * k)
        att.append([1] * len(b["input_ids"]) + [0] * k)
    return (torch.tensor(ids), torch.tensor(labs), torch.tensor(att))


def load_model(cfg, lora_r: int, lora_alpha: int, dropout: float):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False           # incompatible with gradient checkpointing
    peft_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_cfg)
    # ORDER MATTERS: enable checkpointing AFTER the peft wrap, or the wrapper does not
    # inherit it and activations are kept for every layer. Measured 2026-08-26: enabling
    # it before the wrap gave 24.2 GiB peak at batch size 1 on a 3B model and OOM'd at
    # batch 4 on an 80 GiB A100 — obviously wrong, which is why the tuner prints peak
    # memory instead of just reporting the largest batch that happened to fit.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()       # frozen base + checkpointing needs this
    model.print_trainable_parameters()
    return model, tok


def loss_fn(model, ids, labs, att):
    """Causal-LM loss over graded positions only. Returns (SUM over graded tokens,
    n_graded) — a sum, not a mean, so gradient accumulation can normalise exactly.

    Shift by one BY HAND rather than passing `labels=` to the model: the shift is where
    off-by-one masking bugs live, and this way the alignment is visible in the code that
    owns the masks. Position i predicts token i+1, so labels are shifted left.
    """
    out = model(input_ids=ids, attention_mask=att)
    logits = out.logits[:, :-1, :]
    target = labs[:, 1:].reshape(-1)
    flat = logits.reshape(-1, logits.size(-1))
    # Chunk the cross-entropy. Qwen2.5's vocab is ~152k, so a full float32 copy of
    # [B, T, V] is B*T*152k*4 bytes — 5 GB at batch 4 x 2k tokens, plus as much again
    # inside cross_entropy. Chunking caps that at a fixed slice regardless of batch size.
    # Summed over graded tokens and divided once, so the result is identical to computing
    # it in one go (a per-chunk mean would silently weight short chunks too heavily).
    total = flat.new_zeros((), dtype=torch.float32)
    n_graded = int((target != IGNORE_INDEX).sum().item())
    CH = 4096
    for i in range(0, flat.size(0), CH):
        total = total + torch.nn.functional.cross_entropy(
            flat[i:i + CH].float(), target[i:i + CH],
            ignore_index=IGNORE_INDEX, reduction="sum")
    return total, n_graded


def tune_batch(cfg, data, tok, args) -> None:
    """Find the largest per-device batch that fits, at the WORST-CASE sequence length."""
    lens = sorted(len(x) for x in data["input_ids"])
    p50, p99, mx = lens[len(lens)//2], lens[int(len(lens)*0.99)], lens[-1]
    probe_len = mx        # worst case that will ACTUALLY occur, not p99
    total, free = torch.cuda.mem_get_info()[1], torch.cuda.mem_get_info()[0]
    print(f"GPU: {torch.cuda.get_device_name(0)}  {total/2**30:.0f} GiB total, "
          f"{free/2**30:.0f} GiB free")
    print(f"sequence lengths: p50={p50}  p99={p99}  max={mx}")
    print(f"probing at max={probe_len} tokens — the longest example in the dataset.\n"
          f"Tuning on the mean (or even p99) OOMs partway through an epoch when the "
          f"longest example lands, which is the annoying kind of failure.\n")

    model, _ = load_model(cfg, args.lora_r, args.lora_alpha, args.lora_dropout)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    print(f"{'batch':>6}{'peak GiB':>11}{'% card':>9}{'s/step':>9}{'tok/s':>10}  status")
    print("-" * 61)
    results = []
    for bs in [1, 2, 4, 8, 16, 32]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, 1000, (bs, probe_len), device="cuda")
            labs = ids.clone()
            labs[:, : probe_len // 2] = IGNORE_INDEX      # realistic ~50% masked
            att = torch.ones_like(ids)
            t0 = time.time()
            for _ in range(3):
                loss = loss_fn(model, ids, labs, att)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 3
            peak = torch.cuda.max_memory_allocated() / 2**30
            pct = 100 * peak / (total / 2**30)
            tps = bs * probe_len / dt
            results.append((bs, peak, pct, tps))
            print(f"{bs:>6}{peak:>11.1f}{pct:>9.0f}{dt:>9.2f}{tps:>10.0f}  ok")
            del ids, labs, att, loss          # free before the next, larger probe
            if pct > 85:
                print(f"       (stopping: >85% of card)")
                break
        except torch.OutOfMemoryError:
            print(f"{bs:>6}{'—':>11}{'—':>9}{'—':>9}{'—':>10}  OOM")
            break

    # Pick on THROUGHPUT WITH HEADROOM, not "largest that fit". The largest that fits is
    # usually both slower per token (the loss pipeline dominates at this vocab size) and
    # one long batch away from an OOM. Cap at 70% of the card.
    safe = [r for r in results if r[2] <= 70] or results[:1]
    best_bs, peak, pct, tps = max(safe, key=lambda r: r[3])
    largest = results[-1][0]
    print(f"\nRECOMMENDED: --batch-size {best_bs}   ({peak:.0f} GiB, {pct:.0f}% of card, "
          f"{tps:.0f} tok/s)")
    if best_bs != largest:
        big = [r for r in results if r[0] == largest][0]
        print(f"  (batch {largest} also fits at {big[2]:.0f}% but is {big[3]:.0f} tok/s — "
              f"slower per token AND no headroom; not worth the OOM risk)")
    accum = max(1, round(16 / best_bs))
    print(f"  suggested --grad-accum {accum} -> effective batch {best_bs*accum}")
    print(f"  {len(lens)} examples -> {math.ceil(len(lens)/(best_bs*accum))} "
          f"optimizer steps/epoch")


def evaluate_loss(model, loader, dev) -> float:
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for ids, labs, att in loader:
            sl, ng = loss_fn(model, ids.to(dev), labs.to(dev), att.to(dev))
            tot += sl.item(); n += ng
    model.train()
    return tot / max(1, n)          # token-weighted, matching the training objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--tune-batch", action="store_true")
    ap.add_argument("--overfit", type=int, default=0,
                    help="sanity: train on N examples only; loss must approach 0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--run-name", default="sft")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = dr_config.Config.cloud_preset()
    data = torch.load(args.dataset, weights_only=False)
    print(f"dataset: {len(data['input_ids'])} examples from {args.dataset}")
    print(f"  built with: {data.get('config')}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if args.tune_batch:
        tune_batch(cfg, data, tok, args)
        return

    ids_all, labs_all = data["input_ids"], data["labels"]
    if args.overfit:
        ids_all, labs_all = ids_all[: args.overfit], labs_all[: args.overfit]
        print(f"\nOVERFIT SANITY on {len(ids_all)} examples — loss must fall toward 0. "
              f"If it does not, the loss/masking wiring is wrong and no real run will help.")

    idx = list(range(len(ids_all))); random.Random(args.seed).shuffle(idx)
    n_val = 0 if args.overfit else max(1, int(len(idx) * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    mk = lambda ii: TrajDataset([ids_all[i] for i in ii], [labs_all[i] for i in ii])
    coll = lambda b: collate(b, pad_id)
    tr = DataLoader(mk(tr_idx), batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    va = (DataLoader(mk(val_idx), batch_size=args.batch_size, collate_fn=coll)
          if n_val else None)
    print(f"  train {len(tr_idx)} / val {len(val_idx)}   "
          f"batch {args.batch_size} x accum {args.grad_accum} "
          f"= effective {args.batch_size*args.grad_accum}")

    model, _ = load_model(cfg, args.lora_r, args.lora_alpha, args.lora_dropout)
    dev = "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(tr) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total_steps),
        pct_start=args.warmup, anneal_strategy="cos")

    # An overfit check with too few optimizer steps FAILS for the wrong reason and looks
    # like a wiring bug. Measured: 16 examples at effective batch 16 gives 1 step/epoch —
    # 6 steps total, with the LR schedule annealed to 8e-10 by step 5. It reported FAIL
    # while the wiring was fine. Refuse to run rather than produce a misleading verdict.
    if args.overfit and total_steps < 20:
        raise SystemExit(
            f"overfit check would only take {total_steps} optimizer steps "
            f"({len(tr)} micro-batches / accum {args.grad_accum} x {args.epochs} epochs)."
            f"\nThat is too few to converge, and a FAIL would mean nothing. Lower "
            f"--grad-accum (try 2) or raise --epochs so total_steps >= 20.")

    outdir = OUTDIR / args.run_name; outdir.mkdir(parents=True, exist_ok=True)
    hist, step, t0 = [], 0, time.time()
    model.train()
    for ep in range(args.epochs):
        # EXACT token-level normalisation across the accumulation window.
        # Each micro-batch backwards its SUM of losses; the accumulated gradient is then
        # divided ONCE by the total number of graded tokens in the window, just before
        # the step. The obvious `(loss / grad_accum).backward()` with a per-batch mean is
        # a MEAN OF MEANS: every micro-batch gets equal weight regardless of how many
        # graded tokens it holds. That is wrong here because graded-token counts vary a
        # lot (2-turn vs 7-turn trajectories, and tier B drops the answer turn), so short
        # trajectories' tokens would be weighted more than long ones'. Silent, and it
        # skews what the model learns toward short episodes.
        win_loss, win_tok = 0.0, 0
        run_loss, run_tok = 0.0, 0
        for i, (ids, labs, att) in enumerate(tr):
            sum_loss, n_graded = loss_fn(model, ids.to(dev), labs.to(dev), att.to(dev))
            sum_loss.backward()                       # gradient of the SUM
            win_loss += sum_loss.item(); win_tok += n_graded
            run_loss += sum_loss.item(); run_tok += n_graded
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(tr):
                if win_tok:
                    inv = 1.0 / win_tok
                    for prm in params:
                        if prm.grad is not None:
                            prm.grad.mul_(inv)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                win_loss, win_tok = 0.0, 0
                step += 1
                if step % 5 == 0:
                    print(f"  ep{ep} step {step}/{total_steps}  "
                          f"loss {run_loss/max(1,run_tok):.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s",
                          flush=True)
                    run_loss, run_tok = 0.0, 0
        rec = {"epoch": ep, "step": step,
               "train_loss": run_loss / max(1, run_tok) if run_tok else
                             (hist[-1]["train_loss"] if hist else float("nan"))}
        if va:
            rec["val_loss"] = evaluate_loss(model, va, dev)
            print(f"  [epoch {ep}] val_loss {rec['val_loss']:.4f}")
        hist.append(rec)
        model.save_pretrained(str(outdir / f"epoch{ep}"))

    model.save_pretrained(str(outdir / "final"))
    (outdir / "history.json").write_text(json.dumps(
        {"history": hist, "args": vars(args),
         "dataset_config": data.get("config")}, indent=2))
    print(f"\nadapter -> {outdir/'final'}")
    print(f"history -> {outdir/'history.json'}")
    if args.overfit:
        first, last = hist[0]["train_loss"], hist[-1]["train_loss"]
        verdict = ("PASS" if last < first * 0.3 else
                   "FAIL — loss did not collapse; check the masking/loss wiring "
                   "before spending a real run")
        print(f"\nOVERFIT CHECK: {first:.4f} -> {last:.4f}  {verdict}")


if __name__ == "__main__":
    main()
