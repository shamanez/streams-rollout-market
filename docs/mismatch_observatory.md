# Rollout Mismatch Observatory

The Mismatch Observatory is the public proof that this repository needs to exist.

It answers:

> When a rollout comes from a remote endpoint or heterogeneous worker, is it still materially usable as behavior-policy data for a policy-gradient update?

## Demo ladder

### A. Endpoint identity gap

Use free or cheap OpenAI-compatible endpoints to check what the endpoint actually exposes:

- sampled token logprobs
- top logprobs
- token IDs
- seed control
- model revision / checkpoint digest
- tokenizer hash or tokenizer identity
- raw response hash

This is an awareness demo, not a rigorous policy-gradient benchmark, because the exact checkpoint and serving stack are usually not controlled.

### B. Controlled dense mismatch lab

Host the same small dense checkpoint in controlled conditions:

- Transformers reference forward
- vLLM serving path
- SGLang serving path
- BF16 / FP16 / FP8 where supported
- identical prompts, seeds, sampling config, tokenizer

Record rollout tokens and behavior logprobs, then recompute trainer logprobs and compute policy-gradient materiality metrics.

### C. Small MoE router mismatch lab

Use a small open MoE such as Qwen1.5-MoE-A2.7B or another model with accessible routing internals.

Record:

- rollout expert ids by layer/token
- training-side expert ids by layer/token
- router flip rate
- token expert disagreement rate
- ESS before/after router replay
- clipped fraction before/after router replay

This should be the wow demo, but not the first demo.

### D. Market simulation

Run fake and real workers with varied:

- policy lag
- precision class
- engine family
- device class
- context limit
- preemption behavior
- missing metadata

The UI should show that OPBC accepts, corrects, quarantines, or rejects groups with reasons.

## Metrics contract

Every experiment should emit JSON with:

- run_id
- model_id
- checkpoint_digest
- tokenizer_hash
- rollout_engine
- trainer_engine
- precision_class
- quantization_class
- prompt_id
- num_policy_tokens
- delta_logprob_mean
- delta_logprob_abs_mean
- sequence_log_ratio
- ess
- second_moment
- clipped_fraction
- veto_fraction
- max_abs_log_ratio
- top_1pct_gradient_mass
- group_decision
