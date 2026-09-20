# Batch-size tuning for LoRA SFT on an A100 — what we measured, and one open anomaly

**Date:** 2026-08-26. **Model:** Qwen2.5-3B-Instruct + LoRA r=32 on all attention + MLP
projections (59.9M trainable, 1.9%). **Card:** A100-SXM4-80GB (79 GiB usable).
**Status:** the tuner works and the chosen setting is sound. **One thing is unexplained
and is left as an open exercise — see §5.**

---

## 1. The answer we shipped

| batch | peak | % of card | s/step | tok/s | |
|---|---|---|---|---|---|
| 1 | 26.1 GiB | 33% | 0.60 | 3,753 | |
| **2** | **46.2 GiB** | **58%** | **0.86** | **5,195** | ← chosen |
| 4 | OOM at max length | | | | |

Run config: `--batch-size 2 --grad-accum 8` (effective batch 16). Probed at the dataset's
**longest** sequence, 2,240 tokens.

---

## 2. Three bugs the tuning found, none of which were about batch size

Each was found by **printing peak memory** rather than just recording the largest batch
that happened to survive. That distinction is the whole lesson of this document.

### 2.1 Gradient checkpointing was never active — order matters with peft

```python
model.gradient_checkpointing_enable()      # WRONG: before the wrap
model = get_peft_model(model, peft_cfg)
```

The peft wrapper does not inherit it. Symptom: **24.2 GiB at batch size 1** on a 3B LoRA
and an OOM at batch 4 on an 80 GiB card — obviously wrong for the model size, which is the
only reason it was caught. Fix: enable it **after** `get_peft_model`, plus
`model.enable_input_require_grads()` (a frozen base with checkpointing needs it).

Verified rather than assumed afterwards: `is_gradient_checkpointing=True`, weights bf16,
6.0 GiB resident, `use_cache=False`.

### 2.2 The loss materialised the full logits tensor in float32

```python
cross_entropy(logits.reshape(-1, V).float(), ...)   # V = 151,936
```

At batch 4 × 2,240 tokens that is 1.36 **billion** elements: 2.5 GiB in bf16, 5.1 GiB for
the float32 copy, and roughly as much again inside `cross_entropy` — **~13 GiB for the
loss, against 6 GiB for the entire model.**

Chunked into 4,096-row slices, summed over graded tokens and divided **once**. A per-chunk
*mean* would silently overweight short chunks — the same class of error as §2.3.

> **Caveat, measured later (§5):** chunking helped less than expected, because every
> chunk's float32 copy is still retained in the autograd graph for the backward pass.
> Chunking bounds the *peak of a single slice*, not the total.

### 2.3 The tuner was wrong in the way its own docstring warned about

It probed at **p99 = 2,038** tokens while the dataset's max is **2,240**, and recommended
batch 4 at 75/79 GiB. Re-probing at the true max, **batch 4 OOMs.**

That is the worst kind of failure: it dies *partway through an epoch*, when the one long
example finally lands, after the run has been going for a while.

**Rule: tune at the longest sequence that will actually occur, never the mean or p99.**

### 2.4 …and it picked the wrong criterion

"Largest batch that fits" is not the right objective. Here batch 4 was both **slower per
token** than batch 2 (5,190 vs 5,195 tok/s — the big-vocab loss dominates, so the usual
"bigger batch = better utilisation" intuition inverts) **and** one long batch away from an
OOM at 95% of the card.

The tuner now caps at 70% of the card and picks on **throughput with headroom**.

---

## 3. Gradient accumulation — a separate bug in the same area

Harpreet: *"you're gonna use gradient accumulation then?"*

The textbook line is wrong for a token-level loss:

```python
(loss / grad_accum).backward()      # loss is already a per-batch MEAN
```

That is a **mean of means**: every micro-batch gets equal weight regardless of how many
graded tokens it holds. It matters here because graded-token counts vary a lot — 2-turn
versus 7-turn trajectories, and tier-B examples drop the answer turn entirely. Short
trajectories' tokens would be weighted more, biasing the model toward short episodes.

**Correct version:** each micro-batch backwards its **sum**; the accumulated gradient is
divided **once** by the total graded tokens in the window, just before the step.

```python
sum_loss.backward()                 # gradient of the SUM
win_tok += n_graded
...
for p in params: p.grad.mul_(1.0 / win_tok)   # exact token-level mean
opt.step()
```

---

## 4. The overfit check that failed for the wrong reason

After the accumulation fix, the sanity check reported **FAIL**. The wiring was fine: 16
examples at effective batch 16 gives **one optimizer step per epoch** — 6 steps total, with
the LR schedule annealed to 8e-10 by step 5. It never had a chance to converge.

That is the kind of result that sends you debugging a healthy system. The trainer now
refuses an overfit check with fewer than 20 optimizer steps and says what to change.

Properly configured: **0.679 → 0.0066. PASS.** (The check matters — the model can only
drive graded tokens to ~zero if the labels are aligned, so it is the cheap proof that
makes a real run worth spending.)

---

## 5. OPEN ANOMALY — left as a separate exercise

**46 GiB for a 3B LoRA at batch 2 × 2,240 tokens is abnormally high.** Conventional
expectation for this configuration would be batch 8–16 fitting comfortably on an 80 GiB
card. Two hypotheses were tested and **neither accounts for it**:

| hypothesis | test | result |
|---|---|---|
| activations dominate | checkpointing ON vs OFF, batch 2 | **45.8 vs 46.2 GiB** — almost no difference, so activations are NOT the driver |
| the 152k-vocab logits dominate | Liger fused linear+CE (never materialises logits) | **46.2 → 41.0 GiB**; a real 5 GiB saving, but still OOM at batch 4 |

Weights are 6.0 GiB resident. So **~40 GiB is going somewhere not yet isolated.**

Note the two results are in tension with each other: checkpointing-off costing nothing
says activations are negligible, which points at the loss — but removing the logits
materialisation only recovered 5 GiB. Both hypotheses cannot be simultaneously the small
one. Something in the accounting is wrong.

### Where to look next

- `torch.cuda.memory_._record_memory_history()` + `_dump_snapshot()`, then the PyTorch
  memory viewer. This attributes every allocation to a stack — it would answer the whole
  question in one run and is the obvious first move.
- `model.enable_input_require_grads()` makes the **embedding** output require grad. Qwen's
  `embed_tokens` is [151936, 2048]; check whether a gradient is being materialised for it
  despite the base being frozen.
- Whether transformers is casting logits to float32 internally *in addition to* our own
  `.float()`, so the copy exists twice.
- Whether the retained per-chunk float32 copies in the chunked loss are the dominant term
  after all (chunking bounds peak-per-slice, not total-in-graph). A `torch.utils.checkpoint`
  around the loss computation would test this directly.
- Fragmentation: compare `max_memory_allocated` against `max_memory_reserved`.

### Why it is worth an hour

If it is a real bug, the SFT run gets several times faster **and** the same fix likely
helps the GRPO rollout/update stage, which runs for hours rather than minutes. The
current setting is safe and correct — this is upside, not a blocker.

---

## 6. The transferable rules

1. **Print peak memory, don't infer it from what survived.** All three §2 bugs were
   invisible to "did it OOM?" and obvious from one number.
2. **Tune at the longest sequence, not the mean or p99.** Otherwise it fails late.
3. **Largest-that-fits is the wrong objective.** Pick on throughput with headroom; with a
   large vocab, the biggest batch can be slower per token.
4. **With peft, enable gradient checkpointing AFTER the wrap.**
5. **Token-level losses need token-level accumulation**, not a mean of means.
6. **A sanity check needs enough optimizer steps to be meaningful**, or a FAIL tells you
   nothing.
7. **Sanity-check the absolute number against what the model size implies.** 24 GiB (and
   then 46 GiB) for a 3B LoRA should look wrong on sight — that instinct caught bug §2.1
   and is what flagged the anomaly in §5.
