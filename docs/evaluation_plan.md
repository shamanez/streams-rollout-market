# Evaluation plan

## Core metrics

- Accepted rollout tokens/sec.
- Accepted groups/hour.
- Trainer idle fraction.
- Policy lag distribution.
- Reload ACK latency.
- ESS and E[w^2].
- Clipped and veto fraction.
- Rejected group fraction by reason.
- Tokens/sec/$.
- Task pass rate or reward improvement per wall-clock hour.

## First benchmarks

1. Synthetic long trajectory benchmark for K-vs-L sweeps.
2. Local vLLM/SGLang mismatch measurement on a small model.
3. Remote worker churn simulation.
4. Full checkpoint vs patch transport microbenchmark.
