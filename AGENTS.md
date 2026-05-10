# AGENTS.md

This repository implements a decentralized rollout marketplace and rollout validity layer for agentic RL.

It does not implement a trainer.

Do not add PPO, GRPO, RLOO, DPO, FSDP, Megatron, optimizer, gradient-step, or reward-model-training logic except as mocks under tests/ or examples/.

The core product is:
- worker registration and leases
- rollout dispatch
- policy versioning and weight-sync metadata
- LiveStore
- token/logprob contracts
- rollout verification
- off-policyness and training-inference mismatch telemetry
- trainer-facing client APIs
- Mismatch Observatory experiments

All accepted rollouts must preserve:
- token IDs
- sampled-token behavior logprobs
- tokenizer hash
- policy version
- checkpoint digest
- inference engine fingerprint
- precision and quantization class
- tool-token masks
- group membership

GRPO groups must be one-policy-snapshot by default. Mixed-version groups are invalid unless an explicit experimental contract is used.

Before changing contracts, update docs, fixtures, and tests in the same PR.
