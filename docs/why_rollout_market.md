# Why a rollout market must exist

A rollout market is not justified by cheap tokens alone. It is justified by the fact that agentic RL needs a large, variable-latency supply of trajectories, while policy-gradient training only benefits from trajectories whose behavior-policy identity is known well enough to train on.

The repository therefore sells a narrower idea:

> The unit of supply is a verified, policy-pinned, mismatch-accounted trajectory group, not a raw completion.

## Pain points

1. Long-horizon agentic rollouts starve trainers.
2. Remote workers introduce staleness, tokenization drift, precision drift, engine mismatch, device/kernel differences, quantization drift, and MoE routing mismatch.
3. GRPO groups need a coherent behavior policy. By default, every sibling trajectory in a group must share one policy version, tokenizer hash, sampling config hash, precision class, and engine contract.
4. A trainer should not know where a rollout came from. It should consume groups by contract.
5. The market should score quality-adjusted trainable experience, not raw generated tokens.

## Go / no-go

Go if the project ships Mismatch Observatory + OPBC + worker/trainer clients.

No-go if the project becomes a generic queue for cheap GPUs.
