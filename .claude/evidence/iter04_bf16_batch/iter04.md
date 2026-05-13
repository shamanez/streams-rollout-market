# Iter 04 (cycle FP8) — 6 bf16 MoE trajectories captured

(Originally listed as iter 3 in the plan; queue was re-sorted in iter
02 so the FP8 batch ran first and bf16 second.)

## Directive

Drive the bf16 MoE through the capture proxy on port 8001 over the
same six tasks as iter 3, so the matrix has matched rollouts in
both precisions.

## What I did

1. Killed the FP8 vLLM and its worker children (`pkill -9 -f
   Qwen3-30B-A3B-FP8` + explicit GPU pid kill, ~5 s).
2. Launched `vllm_serve_moe_dev.sh` (the bf16 launcher unchanged
   from cycle-3 v2). `/v1/models` returned 200 in ~2 min and
   advertised `Qwen/Qwen3-30B-A3B`.
3. The capture proxy from iter 3 was kept running on port 8001 — no
   restart needed. Sidecar `/tmp/hermes_probes.jsonl` truncated to
   start fresh.
4. Looped `minimal_agent_driver.py` over the same six task IDs:

   | task_id                       | source           | api_calls | wall  |
   |-------------------------------|------------------|----------:|------:|
   | math-tokens-per-gpu-day       | hermes           | 2         | 0.8s  |
   | code-fix-factorial            | hermes           | 3         | 1.4s  |
   | mixed-research-and-math       | hermes           | 5         | 2.0s  |
   | plan-3step-experiment         | hermes           | 3         | 1.3s  |
   | cwc-repo-quality-loop         | cwc_repo         | 2         | 2.1s  |
   | cwc-repo-which-hook-commits   | cwc_repo         | 3         | 1.6s  |

5. Pulled `/tmp/minimal_sharegpt_bf16.jsonl` and
   `/tmp/hermes_probes.jsonl` back to the laptop, merged into
   `runs/live/agent/hermes/qwen3-30b-a3b/bf16/<task_id>.json`. The
   merge overwrote the two cwc-repo trajectories captured in a prior
   session; the no-op-trivia trajectory was untouched and remains
   in the directory.
6. Built `/tmp/rollouts_bf16.json` (19 probe-bearing turns across
   7 trajectories — 6 from this iter plus no-op-trivia) and rsynced
   to the spot for iters 5+6.

## Acceptance evidence

Final bf16 trajectory directory:

```text
$ ls runs/live/agent/hermes/qwen3-30b-a3b/bf16/
code-fix-factorial.json
cwc-repo-quality-loop.json
cwc-repo-which-hook-commits.json
math-tokens-per-gpu-day.json
mixed-research-and-math.json
no-op-trivia.json
plan-3step-experiment.json
```

```text
[hermes-dense] 7 trajectories -> 19 probe-bearing turns (0 skipped)
```

All seven trajectories have at least one assistant turn with
`response_routed_experts` populated; the rollouts payload pushed to
the spot carries 19 entries, one per probe-bearing assistant turn.

## Comparison vs FP8 batch

| Task                          | FP8 api_calls | bf16 api_calls |
|-------------------------------|--------------:|---------------:|
| math-tokens-per-gpu-day       | 2             | 2              |
| code-fix-factorial            | 3             | 3              |
| mixed-research-and-math       | 4             | 5              |
| plan-3step-experiment         | 2             | 3              |
| cwc-repo-quality-loop         | 2             | 2              |
| cwc-repo-which-hook-commits   | 3             | 3              |

The two precisions agree on turn structure for 4 of 6 tasks. On
`mixed-research-and-math` and `plan-3step-experiment`, the bf16
rollout took one extra turn — interesting datapoint for the
router-flip-rate downstream: FP8 quantization changed *what tokens
got generated*, not just *which experts routed them*.
