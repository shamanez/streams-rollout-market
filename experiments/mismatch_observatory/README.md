# Mismatch Observatory experiments

This directory contains the first public demos for the project.

## Experiment A: endpoint identity gap

Goal: show that a hosted endpoint is not automatically a valid behavior policy source.

Record whether the endpoint exposes token IDs, sampled-token logprobs, top_logprobs, seed control, tokenizer identity, provider identity, and raw response hash.

## Experiment B: controlled dense mismatch

Goal: show policy-gradient materiality under controlled serving differences.

Run the same checkpoint through a reference forward pass and rollout-serving backend, then compute:

- delta_t = log_pi_t - log_mu_t
- sequence_log_ratio
- ESS
- E[w^2]
- clipped_fraction
- max_abs_log_ratio
- top_1pct_gradient_mass

## Experiment C: small MoE router mismatch

Goal: show that MoE expert routing makes mismatch discontinuous.

Record rollout-side expert ids and training-side expert ids, then compute router_flip_rate and token_expert_disagreement_rate.

## Experiment D: market simulation

Goal: show OPBC decisions over heterogeneous workers.

Workers should vary by policy lag, precision, engine, context length, and missing metadata.
