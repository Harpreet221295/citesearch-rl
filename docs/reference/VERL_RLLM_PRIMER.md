# verl & rLLM — a first-principles primer

> A saved tutor session. Written page-by-page, story-style, in response to a set of
> questions about **verl (HybridFlow)**: what it is, how it relates to rLLM / vLLM /
> SGLang, single- vs multi-GPU rollout+training, and how it syncs weights between the
> trainer and the rollout engine.
>
> Companion to `RUNPOD_PLAYBOOK.md`. Context: written while deciding to migrate
> `assignments/finqa_agent` (and build `deep_research`) off the hand-rolled HuggingFace
> harness onto **rLLM (agent layer) + verl (engine layer)**.
>
> Sources: [rLLM docs](https://docs.rllm-project.com/) ·
> [verl GitHub](https://github.com/verl-project/verl) ·
> [HybridFlow paper (arXiv:2409.19256)](https://arxiv.org/abs/2409.19256) ·
> [verl docs](https://verl.readthedocs.io/)

---

## Page 1 — verl is the engine rLLM already stands on

You learned **rLLM** first. The plot twist: you were already using verl the whole time —
you just didn't see it. Think of it as two layers stacked on top of each other:

```
   ┌─────────────────────────────────────┐
   │  rLLM   (the AGENT layer)            │
   │  • you define the agent + environment│
   │  • it runs the agent, collects       │
   │    trajectories (obs→action→reward)  │
   └───────────────┬─────────────────────┘
                   │  hands trajectories down
                   ▼
   ┌─────────────────────────────────────┐
   │  verl   (the ENGINE layer)           │
   │  • takes those trajectories          │
   │  • runs the actual GRPO/PPO update   │
   │  • drives the GPUs: generate + train │
   └─────────────────────────────────────┘
```

rLLM's own docs say it plainly: *rLLM handles agent trajectory generation with its
AgentExecutionEngine, then transforms those trajectories into verl's format for model
training using FSDP or Megatron.* So **rLLM is the choreographer; verl is the muscle.**

verl (**V**olcano **E**ngine **R**einforcement **L**earning) is the open-source release
of a ByteDance research system called **HybridFlow**. Its whole job: run the heavy,
distributed, multi-GPU RL post-training loop *efficiently* — the part where your model
has to both **generate** rollouts and **train** on them, over and over. The reason the
finqa run crawled at ~170s/step on raw HuggingFace is exactly the layer verl exists to kill.

---

## Q: "so veRL is more like a trainer?"

### Page 2 — verl is bigger than a trainer; it's an orchestrator

A plain **trainer** (HuggingFace `Trainer`, or the finqa `train.py` update step) does one
job: take a batch, compute loss, step the optimizer. That's the *last* box in the RL loop.

But an RL post-training loop has **four** heavy jobs every step:

```
   ┌─ 1. GENERATE ──────────────┐   the policy rolls out answers
   │    (rollout engine)        │   → this is the slow, autoregressive part
   └────────────┬───────────────┘
                ▼
   ┌─ 2. SCORE ─────────────────┐   reward model / verifier / judge
   └────────────┬───────────────┘
                ▼
   ┌─ 3. REFERENCE forward ─────┐   frozen ref model, for the KL penalty
   └────────────┬───────────────┘
                ▼
   ┌─ 4. TRAIN ─────────────────┐   ← THIS is the "trainer" part
   │    (FSDP / Megatron)       │      compute GRPO loss, optimizer step
   └────────────────────────────┘
```

A trainer only does box 4. **verl orchestrates all four** — it decides which GPUs run
generation, which run the gradient update, how data flows between them, and (the hard
part) how to move freshly-updated weights from box 4 back into box 1's engine.

Sharper one-liner: **verl is an RL post-training *orchestrator* that happens to contain a
trainer.** The trainer is one organ, not the whole body. That "move weights from box 4
back to box 1" arrow is the **weight-sync** problem.

---

## Page 3 — HybridFlow: single-controller + multi-controller

The "hybrid" in HybridFlow is a real design choice. When you coordinate many GPUs there
are two classic styles:

**① Single-controller** — one brain gives orders.
```
        🧠 controller
       ╱    │    ╲
     GPU   GPU   GPU     "you generate, you score, you train"
```
Easy to *read* — the whole RL algorithm is one Python script. But if that one brain
micro-manages every tensor on every GPU, the dispatch overhead becomes a bottleneck.

**② Multi-controller** — every GPU runs the same program, no central boss.
```
   GPU🧠  GPU🧠  GPU🧠     each runs identical code, syncs via NCCL
```
Blazing fast for heavy compute (how normal distributed training works). But *rigid* —
expressing "generate, THEN score, THEN train" across separate models gets tangled.

**HybridFlow's insight:** use *both*, at different levels.
```
  single-controller  →  the RL ALGORITHM   (one script: generate→score→train)
  multi-controller   →  inside each STAGE   (the GPUs grind, no central chatter)
```

So the *dataflow* is one readable script, but each heavy stage runs as flat, fast,
multi-controller distributed compute. PPO-that-reads-like-pseudocode **and** near-peak
throughput — the paper measured **1.5×–20×** over prior frameworks. That two-level split
is also *why* verl can slot different engines into each stage.

---

## Q: "what's NCCL?"

### Page 4 — NCCL and its collective operations

**NCCL** (say "nickel") = **N**VIDIA **C**ollective **C**ommunications **L**ibrary. It's
the plumbing that lets many GPUs talk *directly*, GPU-to-GPU, without bouncing through the
CPU. Its whole job is a handful of **collective operations** — group moves all GPUs do together:

```
  BROADCAST          ALL-REDUCE            ALL-GATHER
  one → everyone     everyone sums up      everyone shares
                     & everyone gets it    their piece

   GPU0: [W]          GPU0:[3] ┐            GPU0:[a]  → [a,b,c,d]
   GPU1: [ ] ← W      GPU1:[1] ├─sum=8→all  GPU1:[b]  → [a,b,c,d]
   GPU2: [ ] ← W      GPU2:[2] │            GPU2:[c]  → [a,b,c,d]
   GPU3: [ ] ← W      GPU3:[2] ┘            GPU3:[d]  → [a,b,c,d]
```

Why it matters for verl:
- **all-reduce** averages gradients across GPUs during the *training* step (data-parallel).
- **broadcast** is one way verl shoves freshly-trained weights *from* the trainer *into*
  the rollout engine — the weight-sync arrow from Page 2.

So "weights are synced via NCCL broadcast" = the top-left box: trainer holds new weights
`[W]`, one collective call sprays them to every rollout GPU at hardware speed.

---

## Q: "what's NVLINK and how is it different from NCCL? and in DP training, are gradients reduced on every GPU and weights updated locally, or updated on one GPU and broadcast?"

### Page 5 — NVLink is the wire, NCCL is the software driving over it

Different layers entirely — they **stack**, they're not alternatives:

```
   ┌────────────────────────────────────────┐
   │  NCCL   — software library              │  "everyone broadcast W"
   │  (the collective ops from Page 4)       │
   └───────────────────┬────────────────────┘
                       │ runs on top of ↓
   ┌────────────────────────────────────────┐
   │  NVLink — physical copper wires         │  actual GPU-to-GPU cables
   │  between GPUs inside one machine        │  ~900 GB/s
   └────────────────────────────────────────┘
```

- **NVLink** = *hardware*: high-speed links NVIDIA solders between GPUs in the same box,
  so GPU0 reads GPU1's memory far faster than via PCIe/CPU. A private highway between GPUs.
- **NCCL** = *software*: when you call "all-reduce," NCCL picks the best route and pushes
  bytes — over NVLink if present, InfiniBand across machines, or PCIe otherwise.

On a single A100 (the finqa case) there's one GPU, so no NVLink and no real collective
traffic — everything's local. NVLink and NCCL only start mattering when you go multi-GPU.

### Page 6 — how multi-GPU training keeps weights in sync

The answer to the either/or: it's **reduce on every GPU and update locally**, and the
reason it works is quietly elegant. Classic **data-parallel** step (each GPU holds a full
copy of weights, works on a different slice of the batch):

```
  GPU0        GPU1        GPU2        GPU3
  grad g0     grad g1     grad g2     grad g3     ← different data → different grads
     └───────────┴─── ALL-REDUCE ──┴───────────┘
                      every GPU now holds  ḡ = (g0+g1+g2+g3)/4   ← identical
     ┌───────────┬───────────────┬───────────┐
  W -= lr·ḡ    W -= lr·ḡ    W -= lr·ḡ    W -= lr·ḡ   ← same update, everywhere
```

The trick: after the all-reduce, **every GPU has the identical averaged gradient.** They
all started from identical weights and apply the identical update, so they stay
bit-for-bit in sync **without anyone broadcasting weights.** No master GPU, no weight
broadcast — the all-reduce on *gradients* is enough.

So why does weight-*broadcast* (Page 4) exist? Different arrow. That broadcast isn't
trainer→trainer, it's **trainer → rollout engine** (a separate system). Inside the
trainer, gradient all-reduce keeps everyone synced; to the outside rollout engine you
still ship the new weights over. Two different problems, easy to conflate.

**Twist for verl:** it usually uses **FSDP** (or Megatron), where each GPU holds only a
*shard* of the weights — so it's reduce-scatter + all-gather instead of a plain
all-reduce. Same principle, sharded.

---

## Page 7 — the two engines verl bolts together

verl doesn't use one piece of software for the whole loop — it bolts together **two very
different engines**, because generating and training want opposite things:

```
   ┌─ ROLLOUT ENGINE ─────────────┐     ┌─ TRAINER ENGINE ─────────────┐
   │  vLLM  or  SGLang            │     │  FSDP  or  Megatron          │
   │  job: GENERATE fast          │     │  job: compute gradients      │
   │  • KV-cache, paged attention │     │  • full fwd+bwd, optimizer   │
   │  • prefix caching            │     │  • sharded weights + grads   │
   │  • inference-optimized       │     │  • training-optimized        │
   └──────────────────────────────┘     └──────────────────────────────┘
```

A great **inference** engine (vLLM) only runs forward passes — hoards a KV-cache, packs
sequences cleverly, never allocates gradient memory. A great **training** engine (FSDP)
runs backward passes and holds optimizer state. Neither is good at the other's job. verl's
bet: don't compromise — use the best tool for each and connect them.

This is exactly the fix for the finqa pain: `train.py` used one HuggingFace model for
*both* generation and training, so every rollout turn re-ran a full prefill with no
cross-turn KV reuse (the ~170s/step tax). vLLM keeps that cache alive → the throughput win.

But bolting two engines together creates one obligation: after the trainer updates
weights, the rollout engine holds **stale** weights. Someone must copy fresh weights from
FSDP-land into vLLM-land every step. That copy is the **weight-sync problem**.

---

## Page 8 — why weight-sync is genuinely hard

Three things make it harder than a `copy()`:

**① The two engines store the same weights in different *shapes*.**
```
  TRAINER layout (FSDP)          ROLLOUT layout (vLLM, TP=2)
  GPU0: [rows 0..N/2]            GPU0: [head-cols 0,2,4..]
  GPU1: [rows N/2..N]            GPU1: [head-cols 1,3,5..]
         └──────────── must be RESHARDED ────────────┘
```
FSDP slices weights one way (flattened, split for gradient math); vLLM slices them another
(by attention heads / tensor-parallel columns for inference). Same numbers, different
jigsaw cut. You can't just hand GPU0's shard to GPU0 — you must **reshard**: gather the
full logical weight, re-cut it the way vLLM wants, deliver it. HybridFlow's
**"3D-HybridEngine"** does this with *zero extra memory copies* and minimal communication.

**② It happens every single step.** No checkpoint-to-disk-and-reload (seconds → death by a
thousand steps). Must be an in-GPU-memory transfer, milliseconds.

**③ Memory pressure.** If both engines live on the same GPU (single-A100), the training
weights + optimizer state + vLLM's KV-cache all want the 80 GB at once.

So weight-sync = *reshard the jigsaw + do it every step + without blowing memory.* Two
ways to solve it: **colocated** (single-GPU) and **disaggregated** (multi-GPU).

---

## Page 9 — colocated mode (the single-A100 case)

The trainer and rollout engine live on the *same* GPU(s) and **take turns**:

```
  ONE A100 (80 GB), time-sharing:

  phase A: GENERATE          phase B: TRAIN
  ┌──────────────────┐       ┌──────────────────┐
  │ vLLM active 🟢    │  -->  │ FSDP active 🟢    │  --> (repeat)
  │ FSDP parked 💤    │       │ vLLM parked 💤    │
  └──────────────────┘       └────────┬─────────┘
                                       │ new weights, same GPU
                                       ▼
                              weight-sync via CUDA IPC
```

Because both engines are on the **same physical GPU**, weight-sync barely travels. The new
weights are already in that GPU's memory — verl hands vLLM a **pointer** via **CUDA IPC**
(two processes sharing the same GPU memory address). No network, no NCCL broadcast across
machines. Near-instant.

The real cost isn't bandwidth, it's **memory juggling**: since both engines want the 80 GB,
verl **offloads** whichever is idle — while vLLM generates, optimizer states get pushed
aside; while FSDP trains, vLLM's KV-cache is freed. They never both hold full memory at once.

This is why "single A100 is sufficient — colocated mode; memory is not the constraint,
install/config friction is." One GPU is enough: no second machine, weight-sync is a local
pointer pass, you pay in time-slicing.

---

## Q: "how much latency does parking FSDP / vLLM and bringing them back cost?"

### Page 10 — the turn-taking tax

The parking/unparking tax has three parts per step:
```
  ① weight-sync (reshard + IPC handoff)   FSDP weights → vLLM layout
  ② offload/reload optimizer + params     GPU ⇄ CPU RAM over PCIe
  ③ engine "wake up"                       vLLM re-grabs KV-cache mem
```

Rough orders of magnitude (single A100; depends heavily on model size + LoRA-vs-full):

| cost | LoRA 4B (your case) | full-param 4B |
|---|---|---|
| ① weight-sync | ~0.1–1 s (small adapter/base moves, IPC local) | ~1–3 s (whole model reshards) |
| ② optimizer offload | ~tiny (LoRA optimizer state is MBs) | ~seconds (Adam state ≈ 8× params fp32, PCIe-bound) |
| ③ vLLM wake-up | ~0.1–0.5 s | ~0.1–0.5 s |

For **LoRA** (your case) the transition tax is small — maybe ~1–2 s/step. For
**full-parameter** training it's the painful one: Adam's optimizer state is ~4× the model
in fp32, and shoving it across PCIe (~16–32 GB/s) every step costs real seconds. That's why
people offload aggressively or avoid colocated for big full-param runs.

**Key mental move:** this tax is a *fixed cost per step*, independent of how many rollouts
are in the step. **Amortize it with big batches** — if the transition costs 2s but
generation+training take 60s, it's noise. Same "batch to amortize" lever as finqa.

---

## Q: "tell me more about the amortize lever, I forget this"

### Page 11 — amortizing fixed cost with batch size

Every training step has two kinds of cost:
```
  FIXED cost   — paid once per step, no matter the batch size
                 (weight-sync, optimizer offload, python/dispatch overhead, engine wake-up)

  VARIABLE cost — scales with how many samples you process
                 (generating N rollouts, N forward/backward passes)
```

What you care about is **cost per useful sample**. Watch batch size act on it:
```
  small batch (8 rollouts/step):
     fixed 2s  +  variable 8s   = 10s   for 8 samples  → 1.25 s/sample   (20% wasted)

  big batch (64 rollouts/step):
     fixed 2s  +  variable 64s  = 66s   for 64 samples → 1.03 s/sample   (fixed ≈ 3%)
```

Same 2s fixed tax — spread over 64 samples instead of 8, it stops mattering. You didn't
make the overhead faster; you made it a smaller *fraction* by piling more real work behind
it. That's **amortization**.

**finqa tie-in:** "batched generation is the cost lever" = exactly this. Generating
rollouts one-at-a-time paid GPU-launch + prefill overhead *per rollout*. Batching many
into one call paid those fixed costs **once** per group → throughput jumped, no algorithm
change.

⚠️ The catch: bigger batches need more memory (KV-cache, activations). The real game is
**"largest batch that still fits"** — big enough to bury fixed costs, not so big you OOM.
That tension is why half the finqa tuning was memory-vs-throughput.

---

## Page 12 — disaggregated mode (multi-GPU)

Spend more GPUs to kill the turn-taking tax. The two engines get their **own** GPUs and run
*at the same time*:

```
  ┌── ROLLOUT GPUs ──┐        ┌── TRAINER GPUs ──┐
  │  vLLM  🟢 always │        │  FSDP  🟢 always │
  │  generating      │        │  training        │
  │  GPU0  GPU1      │        │  GPU2  GPU3      │
  └────────▲─────────┘        └────────┬─────────┘
           │                           │ new weights
           └──── NCCL broadcast ◀──────┘  (reshard + ship over the wire)
```

No parking/unparking — nobody offloads, because vLLM never gives its GPUs back to FSDP.
Optimizer state just *stays* resident on the trainer GPUs. The "② optimizer offload" cost
is gone. You trade it for a different cost: weights physically **travel** trainer→rollout
across NCCL (NVLink same box, InfiniBand across boxes) + the same reshard work every step.

| | colocated (single A100) | disaggregated (multi-GPU) |
|---|---|---|
| GPUs | fewer (share) | more (dedicated) |
| weight-sync | local CUDA IPC (cheap) | NCCL over the wire |
| turn-taking tax | yes (offload/reload) | none — runs concurrently |
| best when | GPU-poor, LoRA | GPU-rich, big models, max throughput |

Even hungrier variant: **async / one-step-off-policy** — the trainer doesn't wait for the
rollout engine to finish; it trains on slightly-stale rollouts so both engines run
flat-out, never idle. verl added this in recent releases. More throughput, at the cost of a
little off-policy-ness.

---

## Page 13 — single vs multi-GPU, for both rollout and training

Yes — verl does all four combinations, because "how many GPUs for rollout" and "how many
for training" are *independent dials*:

```
                    TRAINING
                 1 GPU        many GPUs
              ┌────────────┬────────────────┐
   R   1 GPU  │ everything │ rollout small, │
   O          │ on one A100│ train sharded  │
   L          │ (colocated)│                │
   L  ────────┼────────────┼────────────────┤
   O  many    │ generate   │ full disaggreg.│
   U   GPUs   │ parallel,  │ both scaled out│
   T          │ train on 1 │ (max throughput)│
              └────────────┴────────────────┘
```

The knobs verl exposes:
- **`tensor_model_parallel_size`** (rollout side) — split *one* model across N GPUs so a
  model too big for one card can still generate, or generate faster.
- **FSDP/Megatron world size** (trainer side) — how many GPUs shard training weights + optimizer.
- **`n_gpus_per_node` / `nnodes`** — the total pool verl schedules across.

For the single-A100 finqa case: rollout=1, train=1, **colocated** — a first-class mode, not
a hack. verl was built so the *same* config scales from 1 GPU to 1000 by changing these
numbers — you don't rewrite the algorithm, you change a resource dial. **You write the RL
logic once (single-controller), and the GPU count is just config.** That decoupling is the
HybridFlow payoff from Page 3.

---

## Q: "do we control which of fsdp/megatron/vllm it uses?"

### Page 14 — the backends are pluggable config

Those engines are **pluggable backends you select in config**. You mix-and-match a
*training* backend with a *rollout* backend:

```
   TRAINER backend (pick one)        ROLLOUT backend (pick one)
   ┌─────────────────────────┐      ┌─────────────────────────┐
   │  fsdp   (default, easy)  │      │  vllm    (default)       │
   │  fsdp2  (newer, faster)  │  ×   │  sglang  (great for      │
   │  megatron (max scale)    │      │          agents/tools)   │
   └─────────────────────────┘      └─────────────────────────┘
```

In verl's config, roughly:
```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp          # ← fsdp | fsdp2 | megatron
  rollout:
    name: vllm              # ← vllm | sglang
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.6
```

**Trainer backend:**
- **FSDP / FSDP2** — easy default. Shards weights+optimizer across GPUs, minimal config.
  Fine up to ~30–70B. FSDP2 is the newer rewrite (faster, cleaner). *This is what you'd use.*
- **Megatron** — heavy machinery for *huge* models (100B+): tensor/pipeline/expert
  parallelism. More setup pain; only worth it at scale you're nowhere near.

**Rollout backend:**
- **vLLM** — default; paged-attention + prefix caching (your throughput fix).
- **SGLang** — sibling inference engine, especially strong for multi-turn / tool-calling
  agents (RadixAttention reuses KV across branching conversations). Worth remembering for
  finqa / deep-research, which are exactly multi-turn tool agents.

Practical recipe (boring in the best way): **`strategy: fsdp2` + `rollout: vllm`**, single
A100, colocated. Exotic knobs (megatron, big TP) exist for when you outgrow one GPU — flip
a config value, don't rewrite.

---

## Q: "why not recommend SGLang then, even single-GPU colocated?"

### Page 15 — SGLang is a great fit; vLLM is the widest paved road

You're right that SGLang is arguably the *better technical fit*: finqa and deep-research
are multi-turn tool agents, and SGLang's **RadixAttention** is purpose-built to reuse
KV-cache across branching, multi-turn conversations — the exact shape of a ReAct loop. On
the merits it's a legitimate pick, single-GPU colocated included. vLLM isn't recommended
because SGLang is worse.

vLLM is the default for **one** reason — the enemy your own notes flagged: *"memory is not
the constraint, install/config friction is."*
- vLLM is verl's **oldest, most-documented, most-battle-tested** rollout backend. At 2am on
  a rented pod, ~90% of verl issues/tutorials assume vLLM.
- SGLang support in verl is solid but **newer** — more version-matching sensitivity
  (SGLang ↔ verl ↔ torch pins), the kind of friction that eats hours.

**Coaching call:** get *one* backend green first — vLLM, widest paved road — then A/B
SGLang as a deliberate experiment once the harness runs end-to-end. Then you measure the
RadixAttention win on *your* traces instead of taking a recommendation on faith. Starting on
SGLang and eating a bit more setup risk for the better-fit engine is also defensible — your dial.

---

## Page 16 — how rLLM rides on top of verl

Where rLLM (the L7 layer) sits on everything above. The seam, exactly:

```
   ┌─ rLLM ────────────────────────────────────────────┐
   │  you write:  Agent + Environment + reward          │
   │  AgentExecutionEngine runs the agent, collects     │
   │  trajectories:  [obs → action → obs → ... → reward]│
   └───────────────────────┬────────────────────────────┘
                            │  converts trajectories → verl's data format
                            ▼
   ┌─ verl ─────────────────────────────────────────────┐
   │  everything from pages 2–15:                        │
   │  • rollout engine (vLLM/SGLang)  ── generate        │
   │  • trainer (FSDP/Megatron)       ── GRPO update      │
   │  • weight-sync                   ── colocated/disagg │
   └─────────────────────────────────────────────────────┘
```

Division of labor:

| layer | you think about… | you *don't* think about… |
|---|---|---|
| **rLLM** | agent logic, tools, when to stop, reward | GPUs, sharding, weight-sync |
| **verl** | *(config only)* which backends, how many GPUs | — it just runs the loop |

In L7, when you defined an agent and it "just trained," verl was underneath doing the vLLM
generation, the FSDP gradient step, and the weight-sync every step. You were driving; verl
was the engine you never had to open the hood on. The strategic bet: build finqa +
deep-research on **rLLM (agent layer) + verl (engine)** — write the *interesting* part (agent
+ reward) once; the same code scales from a single A100 to a multi-node cluster by changing
config numbers. The throughput pain from the hand-rolled harness lives entirely in the
bottom box now, already solved.

---

## Page 17 — the finqa migration map (what survives, what dies)

```
  YOUR HAND-ROLLED finqa_agent          →   rLLM + verl
  ─────────────────────────────────         ─────────────────────────
  tools.py (lookup_cell, calc, ...)     →   ✅ KEEP  → rLLM Environment
  reward.py (grounded + judge)          →   ✅ KEEP  → rLLM reward fn
  data.py (FinQA loader, difficulty)    →   ✅ KEEP  (feed as dataset)
  the tool-token MASKING concept        →   ⚠️ RE-VERIFY it holds in verl
  ─────────────────────────────────         ─────────────────────────
  rollout.py (batched HF generate)      →   ❌ DELETE → verl's vLLM
  logprobs.py (manual logprob recompute)→   ❌ DELETE → verl handles it
  loss.py (GRPO advantage + clip)       →   ❌ DELETE → verl's GRPO
  train.py (the whole 170s/step loop)   →   ❌ DELETE → verl trainer
  masking.py (manual model_mask build)  →   ⚠️ verl's multi-turn masking
```

Headline: **all the *learning logic you wrote* ports; all the *plumbing that fought you*
gets thrown away.** Your tools, reward, and data — the parts encoding "what is FinQA and
what's a good answer" — become an rLLM Environment + reward function. The rollout loop,
logprob bookkeeping, GRPO math, and the 170s/step training loop become config.

The one thing to **watch carefully**: **tool-token masking** (only gradient-update the
model's own tokens, never tool outputs). verl has its own multi-turn masking mechanism —
confirm it masks the same tokens your `model_mask` did. This is the single most important
correctness check of the migration, because a masking bug is **silent**: training looks
fine, the model learns garbage.

---

### One-line recap

verl is an RL post-training **orchestrator** (rollout engine + trainer + weight-sync)
built on HybridFlow's single-controller-algorithm / multi-controller-compute split. It
runs on one GPU (**colocated**, CUDA-IPC weight-sync, time-sliced) or many
(**disaggregated**, NCCL weight-sync, concurrent). Backends are pluggable
(**FSDP/FSDP2/Megatron** × **vLLM/SGLang**). **rLLM** rides on top as the agent layer; you
write the agent + reward once and scale by config.

---
---

# Part 2 — the rLLM API: plugging in an agentic task and running training

> Second tutor session. Goal: get the *feel* of the basic rLLM syntax — how you set up an
> agentic task and launch training — using a tiny tool-agent example that is structurally
> identical to finqa.
>
> API accuracy grounded in the current rLLM docs & README (Aug 2026): class-based
> `ToolAgent` / `ToolEnvironment` / `AgentTrainer(backend="verl")`. rLLM also has a newer
> decorator API (`@rllm.rollout` / `@rllm.evaluator`, often shown with the Tinker backend);
> this walkthrough uses the class-based path because it's the one the verl-backed examples
> use and it maps cleanly onto finqa.

## Page 1 — the four pieces (mapped to finqa)

Plugging an agentic task into rLLM is **four pieces**, all of which you already built in finqa:

```
   rLLM piece          what it is                finqa equivalent
   ─────────────────────────────────────────────────────────────────
   1. Environment  →   the task + tools + reward    tools.py + reward.py
   2. Agent        →   the policy / ReAct loop      rollout.py's turn loop
   3. reward fn    →   scores a finished episode     reward.py's scorer
   4. AgentTrainer →   runs RL, drives verl          train.py (deleted!)
```

They snap together in a gym-style loop:

```
        ┌──────────────────────────────────────────┐
        │                                            │
        ▼                                            │
   Environment.reset() ──obs──► Agent ──action──► Environment.step(action)
                                  ▲                     │
                                  └──── obs, reward ────┘
                        (repeat until done → episode ends → reward)
```

Two roads to build it:
- 🛣️ **Easy road** — use rLLM's prebuilt `ToolAgent` + `ToolEnvironment`; hand it your tools
  + a reward function. Barely any code.
- 🔧 **Custom road** — subclass `BaseAgent` / `BaseEnv` and write `reset`/`step` yourself.

## Page 2 — the easy road, end-to-end

A complete training script using the prebuilt tool agent (the real `math_tool` example —
structurally finqa):

```python
from rllm.agents import ToolAgent
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.trainer.agent_trainer import AgentTrainer

# 1️⃣ the AGENT — a ReAct tool-caller, configured, not coded
agent_args = {
    "tools": ["python"],                       # ← finqa: ["lookup_cell","calc",...]
    "parser_name": "qwen",                     # how to parse tool calls from output
    "system_prompt": "You solve math problems by writing python.",
}

# 2️⃣ the ENVIRONMENT — holds the tools + the reward function
env_args = {
    "tools": ["python"],
    "reward_fn": math_reward_fn,               # ← finqa: your grounded+judge scorer
}

# 3️⃣ the TRAINER — wires it all to verl and runs GRPO
trainer = AgentTrainer(
    agent_class=ToolAgent,
    env_class=ToolEnvironment,
    agent_args=agent_args,
    env_args=env_args,
    config=config,                             # verl config (model, lr, batch…)
    train_dataset=train_dataset,
    val_dataset=test_dataset,
    backend="verl",                            # ← the whole engine from Part 1
)

trainer.train()                                # 4️⃣ go
```

What's **not** there: no generation loop, no logprob math, no GRPO advantage, no optimizer
step, no weight-sync. All of it is the `backend="verl"` line. The feel of the easy road:
**describe the agent in a dict, describe the env in a dict, hand both to `AgentTrainer`, call
`.train()`.**

## Page 3 — LLM-as-judge served by vLLM (you host it)

The tiny example uses a pure-Python reward (string match). If your reward needs an LLM judge
(finqa does), the clean pattern is a **separate frozen vLLM server** your `reward_fn` calls:

```python
from openai import OpenAI
judge = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")  # a vLLM server

def finqa_reward_fn(task, action, **kwargs):
    grounded = exact_match(action, task["gold"])          # cheap verifiable part
    verdict = judge.chat.completions.create(               # LLM-as-judge part
        model="qwen-judge",
        messages=[{"role": "user", "content": judge_prompt(task, action)}],
    )
    soft = parse_score(verdict.choices[0].message.content)
    return 0.5 * grounded + 0.5 * soft
```

```
   ┌─ policy rollout ─┐   ┌─ judge ─────────┐   ┌─ trainer ─┐
   │ vLLM (TRAINING)  │   │ vLLM (FROZEN)   │   │ FSDP      │
   │ weights change ✏️│   │ weights fixed 🧊│   │ GRPO      │
   └──────────────────┘   └─────────────────┘   └───────────┘
```

**Why a separate frozen server:** the policy's weights change every step; a judge that
drifts every step is a broken ruler and even easier to hack. The judge must be frozen, so it
can't share weights with the training policy.

## Page 4 — rLLM has no built-in judge server (you bring the endpoint)

rLLM gives you the reward *hook*, not the *hosting*:

```
   rLLM gives you:   reward_fn(task, action) -> float     ← a plain Python function
   rLLM does NOT:    spin up / manage a judge LLM for you
```

If your reward needs an LLM judge, you start the vLLM server (or point at an API) yourself
and call it from inside `reward_fn`. rLLM won't launch it, batch it, or keep it frozen.

| want | who provides it |
|---|---|
| plain function reward | ✅ rLLM (`reward_fn`) |
| generative LLM-as-judge | ❌ nobody — you serve it (vLLM sidecar) |
| scalar reward-model worker | ⚙️ verl backend, via config (`reward_model`) — aimed at scalar RMs, not generative judges |

## Page 5 — the custom Environment (`BaseEnv`)

For when `ToolEnvironment` doesn't fit — subclass `BaseEnv`, implement the gym loop:

```python
from rllm.environments.base.base_env import BaseEnv

class FinQAEnv(BaseEnv):
    def __init__(self, task):
        self.task = task            # one question + its tables + gold answer

    def reset(self):
        obs = build_prompt(self.task)        # question + tables_menu
        return obs, {}                       # (observation, info)

    def step(self, action):
        if is_final(action):
            reward = finqa_reward_fn(self.task, action)
            return "", reward, True, {}       # (obs, reward, DONE, info)
        else:
            obs = run_tool(action)            # e.g. lookup_cell(...) result
            return obs, 0.0, False, {}        # not done → reward 0, keep going

    @staticmethod
    def from_dict(env_args):
        return FinQAEnv(task=env_args["task"])   # rLLM builds envs via this
```

The shape to burn in:
```
   reset()  →  (obs, info)
   step(a)  →  (obs, reward, done, info)      ← the classic RL 4-tuple
```

Rewards are 0 until the episode ends (sparse terminal reward). The rLLM-specific quirk:
**`from_dict`** — rLLM re-creates a fresh env per rollout from `env_args` via this static
method, so every custom env needs one.

## Page 6 — can you track grad_norm, KL, dead groups, etc.?

Yes — a big win of moving to verl. verl auto-logs to W&B/console/TensorBoard:

| metric | verl key (roughly) |
|---|---|
| grad norm | `actor/grad_norm` |
| KL to reference | `actor/kl`, `actor/kl_coef` |
| policy-gradient loss | `actor/pg_loss` |
| entropy | `actor/entropy` |
| PPO clip fraction | `actor/clipfrac` |
| reward mean/std/max | `critic/rewards/*` |
| response length | `response_length/mean` |
| learning rate | `actor/lr` |
| timing (gen vs train) | `timing/*` |

- 🎯 **"Dead groups"** is *your* GRPO diagnostic — not a verl default name, but the
  ingredients (per-group reward std) are computed, so it's a small custom metric you add.
- 🤖 **Agent-level metrics** (turns, tool-use, per-difficulty solve) come from the **rLLM**
  layer / whatever you put in the env's `info` dict.

Model: **verl logs optimization health, rLLM logs agent behavior, your `info` dict is the
hook for anything custom.**

## Page 7 — the custom Agent (`BaseAgent`)

The Environment owns the task; the Agent owns the conversation + records the trajectory:

```python
from rllm.agents.base_agent import BaseAgent

class FinQAAgent(BaseAgent):
    def __init__(self):
        self.messages = []
        self._trajectory = Trajectory()

    def update_from_env(self, observation, reward, done, info, **kwargs):
        self.messages.append({"role": "user", "content": observation})

    def update_from_model(self, response, **kwargs):
        self.messages.append({"role": "assistant", "content": response})
        return Action(action=response)      # what gets sent to env.step()

    @property
    def chat_completions(self):
        return self.messages        # what the rollout engine feeds the model next

    @property
    def trajectory(self):
        return self._trajectory     # what verl trains on
```

Per-turn ping-pong:
```
   env.step() ──obs──► agent.update_from_env(obs)      "here's a tool result"
                       agent.chat_completions ──► vLLM generates
        vLLM response ──► agent.update_from_model(resp) "record the action"
                       Action ──► env.step(action)      "run it"
```

The Agent is memory + formatter; the accumulated `trajectory` flows to verl for the GRPO
update. This is finqa's `rollout.py` turn-loop split into named hooks.

## Page 8 — config + launch

Hydra-style: a base YAML + command-line overrides. A single-A100 finqa launch:

```bash
python train_finqa.py \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.model.lora_rank=32 \
  data.train_batch_size=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  trainer.n_gpus_per_node=1 \
  trainer.total_epochs=1 \
  trainer.logger=['console','wandb'] \
  trainer.project_name=finqa_agent
```

The `.sh` script *is* your experiment record — swap `lr`, `lora_rank`, `max_turns`, rerun,
and that one line is the diff between runs.

## Q: what are `train_batch_size` and `ppo_mini_batch_size`?

### Page 9 — the two vLLM instances / multi-LoRA judge question

**Two vLLM instances on one GPU?** Technically yes (split `gpu_memory_utilization`), but
wasteful — two copies of the base weights + two KV pools.

**Better idea: one instance, toggle LoRA.** vLLM natively supports **multi-LoRA serving** —
base loaded once, pick per-request whether to apply an adapter:

```
   ONE vLLM instance, base Qwen-4B loaded once 🧊
        ├── request WITH adapter   → base + policy-LoRA   = the POLICY  ✏️
        └── request WITHOUT adapter → base (no delta)      = the JUDGE  🧊
```

Bonus: LoRA is a separate delta, so base weights never change during LoRA training → "base
with no adapter" is automatically a **frozen judge**.

**How it plays with verl/rLLM:** verl already hot-swaps the policy's LoRA delta (~50 MB,
sub-ms) into its rollout vLLM via `LoRARequest` — that's its weight-sync. The friction is
routing your judge calls into verl's *internally-managed* engine, which needs verl's newer
**async/agent-server mode** (exposes an OpenAI endpoint).

| approach | weight copies | difficulty |
|---|---|---|
| separate judge vLLM server | 2 | 🟢 paved road — do first |
| reuse verl's engine, base-as-judge via no-adapter | 1 | 🟡 needs async server mode |

Ship the separate judge server first; optimize to single-instance multi-LoRA once green.

### Page 10 — the batch hierarchy

Three nested batch sizes:
```
  ┌─ data.train_batch_size = 64 ────────────────────────────────┐
  │  COLLECT: sample 64 prompts, generate all their rollouts     │
  │   ┌─ ppo_mini_batch_size = 16 ─┐  ┌──16──┐ ┌──16──┐ ┌──16──┐ │
  │   │ one OPTIMIZER STEP on 16   │  │ step │ │ step │ │ step │ │
  │   └────────────────────────────┘  └──────┘ └──────┘ └──────┘ │
  │        64 collected / 16 per update = 4 gradient updates      │
  └──────────────────────────────────────────────────────────────┘
```

```
  train_batch_size    → GATHER    how much experience collected per step
  ppo_mini_batch_size → DIGEST    how much consumed per optimizer step
                                  (<full ⇒ multiple updates per gather)
  micro_batch_size    → FIT       memory only, math-neutral (grad-accum)
```

## Q: reconciling with finqa (256 rollouts, grad_accum 8, one step?)

### Page 11 — two reasons to chop a batch (accumulation ≠ multiple updates)

finqa cloud config: `group_size=16`, `prompts_per_step=2`, `grad_accum=8` (original 256-attempt):
```
  prompts_per_step=2 × group_size=16  = one MICRO-BATCH = 32 rollouts
  × grad_accum=8                      = 256 rollouts collected
     compute grads on each of 8, ADD them, then ONE optimizer.step()
```
So: **256 rollouts, 8 accumulation passes, one weight update.** ✅ Correct.

The distinction that trips everyone:
```
  ① GRADIENT ACCUMULATION  (finqa grad_accum=8)
     WHY: memory. Split into chunks, sum grads, ONE step.
     → mathematically IDENTICAL to one big batch.
  ② MINI-BATCH / MULTIPLE UPDATES  (finqa did NOT do this)
     WHY: more updates from the same collected data (PPO's multiple epochs).
     → changes the math (data reused across updates).
```
Memory hook: **accumulation = "split for memory, still one step"; mini-batch = "split to
take more steps."**

### Page 12 — PPO multiple epochs = importance sampling + clip

Multiple updates per batch → after update #1 the policy moved, so the collected data is now
off-policy → weight each sample by the ratio `r = π_new/π_old` and clip it:

```
  collect 256 rollouts with π_old
  update #1 → policy is now π_new (weights moved)
  update #2 uses SAME data (generated by π_old) → OFF-POLICY
  fix: ratio r = π_new/π_old, clipped to [1-ε, 1+ε]
```

Why finqa felt like it needed none of this: **1 update per batch → π_new == π_old → ratio =
1.0 → clip inert → purely on-policy.** Same GRPO loss; the clip only earns its keep once you
take multiple updates per batch.

### Page 13 — why finqa used single-update, on-policy

1. **Correctness-first:** ratio ≡ 1 means the gradient is correct even if the IS/clip code
   had a bug — one fewer silent failure mode for a lab where you wrote the loss.
2. **GRPO convention:** DeepSeek's GRPO defaults to one update per batch (`μ=1`).
3. **Stability trade-off:** rollouts are expensive (argues *for* reuse), but the batch is
   tiny + noisy (argues *against* — hammering 4 updates into one noisy batch overfits/drifts).

Honest note: given finqa's throughput pain, `ppo_epochs > 1` is a legitimate lever to A/B on
verl — more learning per expensive rollout — but watch KL/reward-std and pair with a bigger
batch. It trades on-policy stability for sample-efficiency; not a free win.

### Page 14 — finqa → verl config mapping (the Rosetta stone)

```
  finqa (config.py)                   verl config key                         value
  ──────────────────────────────────────────────────────────────────────────────────
  group_size = 16                  →  actor_rollout_ref.rollout.n            = 16
  prompts_per_step × grad_accum    →  data.train_batch_size (prompts/step)   = 16
  → total 256 rollouts             →  (derived: train_batch_size × n)        = 256
  ONE update per step              →  actor.ppo_mini_batch_size = full  +   1 update
                                       actor.ppo_epochs                       = 1
  grad_accum = 8  (memory split)   →  actor.ppo_micro_batch_size_per_gpu     ≈ 32 seqs
  prompts_per_step = 2 (micro)     →  (micro-batch = 2 prompts × 16)         = 32 seqs
  loss_chunk_size (OOM guard)      →  subsumed by micro_batch_size_per_gpu
  gen_batch_size = 16              →  vLLM handles internally (max_num_seqs)
  lr = 1e-4                        →  actor.optim.lr                         = 1e-4
  lora_rank / alpha                →  actor.model.lora_rank / lora_alpha
  kl_coef = 0.02                   →  actor.kl_loss_coef (+ algorithm.kl_ctrl)
  model + revision                 →  actor_rollout_ref.model.path
  GRPO                             →  algorithm.adv_estimator                = grpo
  max_turns = 6                    →  (rLLM agent-level, not verl)
```

Two takeaways: finqa's `prompts_per_step` **is** verl's micro-batch; and `max_turns` lives in
**rLLM** (agent behavior), while batch/lr/KL/GRPO live in **verl** (optimization).

⚠️ Honesty caveat: verl's exact **unit convention** for `ppo_mini_batch_size` (prompts vs
sequences, and whether it auto-multiplies by `rollout.n`) has shifted across versions —
confirm the unit in the verl docs for your installed version. The concept is above; the exact
integer needs a doc check.

## Page 15 — cheat-sheet

```
  ── THE FOUR PIECES ───────────────────────────────────────────────
  Environment   task + tools + reward     BaseEnv: reset()→obs,
                                          step(a)→(obs,reward,done,info)
  Agent         the conversation/policy   BaseAgent: update_from_env(),
                                          update_from_model(), .trajectory
  reward_fn     scores an episode         plain Python fn (you host any LLM judge)
  AgentTrainer  runs RL via verl          AgentTrainer(...).train()

  ── EASY ROAD (use prebuilt) ──────────────────────────────────────
  agent_class=ToolAgent,  env_class=ToolEnvironment
  agent_args={tools, parser_name, system_prompt}
  env_args  ={tools, reward_fn}
  backend="verl"

  ── WHO OWNS WHAT ─────────────────────────────────────────────────
  rLLM  → agent behavior:  tools, turns (max_turns), reward hook, trajectory
  verl  → optimization:    GRPO, batch/lr/KL, FSDP+vLLM, weight-sync, logging

  ── BATCH HIERARCHY ───────────────────────────────────────────────
  train_batch_size   GATHER   how many prompts collected per step
      × rollout.n              GRPO group size → total rollouts
  ppo_mini_batch     DIGEST   <full ⇒ multiple updates (needs clip + IS)
  micro_batch_gpu    FIT      memory only, math-neutral (= grad-accum)
  ppo_epochs=1                finqa's on-policy default (ratio=1, clip inert)

  ── LAUNCH ────────────────────────────────────────────────────────
  python train.py \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.n_gpus_per_node=1  trainer.logger=[console,wandb]
```

The arc: **describe agent + env + reward → `AgentTrainer(backend="verl")` → `.train()`**, and
everything from Part 1 (rollout engine, GRPO, weight-sync, logging) runs underneath by config.
