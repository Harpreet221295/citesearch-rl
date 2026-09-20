# Engineering log — index

These are the working documents written while the project was built and run. They are
kept verbatim: they record decisions at the time they were made, including conclusions
that later evidence overturned. Read the top-level [`README.md`](../README.md) for the
current state; come here for the *why* and the *how*.

## Start here

| file | what it is |
|---|---|
| [`START_HERE.md`](START_HERE.md) | Status at the end of the last run and the **findings register** (F1–F24, each marked CONFIRMED / KILLED / OPEN). |
| [`RL_FROM_SFT_LOG.md`](RL_FROM_SFT_LOG.md) | The RL-from-SFT session, end to end: design, build, every issue, every result, and the this-run-vs-earlier-runs table. |
| [`../distill/RESULTS.md`](../distill/RESULTS.md) | Every SFT number in one place, three arms, both eval sets. |
| [`HANDOFF.md`](HANDOFF.md) | The exact commands to resume work, plus the history of earlier handoffs below it. |

## By topic

| you want to… | read |
|---|---|
| understand the reward and environment design decisions | [`NOTES.md`](NOTES.md) |
| see how the teacher data was collected and validated | [`../distill/DATA_COLLECTION_LOG.md`](../distill/DATA_COLLECTION_LOG.md) |
| see how SFT was trained and tuned | [`../distill/SFT_HISTORY_LOG.md`](../distill/SFT_HISTORY_LOG.md), [`../distill/BATCH_SIZE_TUNING.md`](../distill/BATCH_SIZE_TUNING.md), [`../distill/SFT_RL_PLAN.md`](../distill/SFT_RL_PLAN.md) |
| understand why the first GRPO runs failed | [`TRAINING_HISTORY_LOG.md`](TRAINING_HISTORY_LOG.md), then finding F18 in `START_HERE.md` |
| read the harness audit that found 8 bugs in the tool-calling format | [`../rft_diagnosis/FORMAT_INVESTIGATION_LOG.md`](../rft_diagnosis/FORMAT_INVESTIGATION_LOG.md), [`../rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md`](../rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md) |
| set up the out-of-distribution eval (MuSiQue) | [`GENERALIZATION_EVAL.md`](GENERALIZATION_EVAL.md) |
| install the rLLM + veRL + vLLM stack on a fresh GPU box | [`FRESH_POD_SETUP_AND_SANITY_CHECK.md`](FRESH_POD_SETUP_AND_SANITY_CHECK.md), [`RLLM_VERL_INSTALL_NOTES.md`](RLLM_VERL_INSTALL_NOTES.md) |
| see how the environment was ported into rLLM's `Workflow` contract | [`WORKFLOW_PORT_NOTES.md`](WORKFLOW_PORT_NOTES.md) |
| tune one training step (throughput, memory, liger, batch shapes) | [`ONE_STEP_TUNING_VERL_RLLM.md`](ONE_STEP_TUNING_VERL_RLLM.md) |
| serve a trained adapter with vLLM or SGLang | [`INFERENCE_SERVING.md`](INFERENCE_SERVING.md) |
| see what was considered and deferred | [`TENTATIVE_FUTURE_EXPERIMENTS.md`](TENTATIVE_FUTURE_EXPERIMENTS.md) |
| the original per-lab worklog template this grew out of | [`ORIGINAL_WORKLOG_README.md`](ORIGINAL_WORKLOG_README.md) |

## Reference material

| file | what it is |
|---|---|
| [`reference/VERL_RLLM_PRIMER.md`](reference/VERL_RLLM_PRIMER.md) | A first-principles primer on how rLLM (agent layer) and veRL (training engine) fit together, written before the port. |
| [`reference/RUNPOD_PLAYBOOK.md`](reference/RUNPOD_PLAYBOOK.md) | The operations manual for cheap, crash-proof rented-GPU runs: batched generation, wall-clock caps, checkpoint/resume mirrored to the Hub, dashboards. |

Superseded and kept for the record only: `POD_LIFECYCLE.md` (the stop/resume era), the older
sections of `HANDOFF.md`, and the A/B table in `rft_diagnosis/RFT_PLAN_AND_MODEL_DIAGNOSIS.md`.
