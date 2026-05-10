# OPBC metrics

OPBC means Off-Policy Budget Controller.

The controller converts rollout telemetry into a decision:

- train
- train_with_correction
- replay
- quarantine
- reject

## Core quantities

For every valid policy token t:

```text
delta_t = log_pi_t - log_mu_t
```

where `mu` is the rollout behavior policy and `pi` is the trainer/current policy.

Masked tokens are excluded:

- tool outputs
- environment text
- padding
- non-policy system/control tokens

Group or segment ratio:

```text
log_ratio = sum(mask_t * delta_t)
w = exp(clip(log_ratio, -C, C))
ESS = (sum_i w_i)^2 / (n * sum_i w_i^2)
```

## Materiality metrics

- ESS
- E[w^2]
- clipped_fraction
- veto_fraction
- max_abs_log_ratio
- top_1pct_gradient_mass
- per-turn ESS
- sequence log-ratio slope vs length
- router_flip_rate for MoE
- token_expert_disagreement_rate for MoE

## Source decomposition

OPBC should log source labels even if the first decision function is simple:

- policy_lag
- wall_clock_lag
- engine_mismatch
- precision_mismatch
- quantization_mismatch
- tokenizer_mismatch
- device_class
- MoE_router_mismatch
- missing_metadata

A single budget score is useful as a guardrail, but source-specific telemetry is mandatory because different mismatch sources have different bias structures.
