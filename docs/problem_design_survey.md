# Decentralizing the rollout layer of streams

## Thesis

For agentic RL, rollout supply is becoming the dominant bottleneck. A useful rollout market cannot be only a queue of GPUs. It must be a versioned, token-level, auditable protocol where staleness, numerical mismatch, quantization drift, device heterogeneity, and weight-transport latency are measured under one off-policyness budget.

## Problem

Synchronous RL keeps data fresh but starves the trainer when rollouts are long. Asynchronous RL keeps workers and trainers busy but creates policy debt. In agentic RL this debt is severe because trajectories are long, tool-using, stochastic, and often generated on heterogeneous inference stacks.

## Design

- Pin one logical behavior policy per GRPO group.
- Store token IDs, masks, rollout logprobs, rewards, policy version, engine, precision, device, quantization, and telemetry.
- Compute OPBC metrics before a group enters the trainer batch.
- Relax global all-or-nothing reload into routing-pool quorum while preserving group-level policy pinning.
- Treat untrusted workers as auditable producers, not as trusted trainer extensions.

## Research agenda

1. K-vs-L frontier for long agentic trajectories.
2. Mismatch decomposition across engine, precision, device, quantization, and staleness.
3. Per-turn or segment-level importance sampling for tool-using trajectories.
4. k-of-n reload quorum with C-0 preservation.
5. Sparse patch transport with quantized inference receivers.
6. Verifiable worker claims using hashes, recompute audits, and receipts.
