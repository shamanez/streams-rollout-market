# Iter 03 (cycle FP8) — 6 FP8 MoE trajectories captured

(Originally listed as iter 4 in the plan; queue was re-sorted in iter
02 so the FP8 batch runs first while the FP8 serve is hot.)

## Directive

Drive the FP8 MoE through the capture proxy on port 8001 over six
tasks (4 from `agent_tasks_hermes.json`, 2 from `agent_tasks_cwc_repo.json`)
and verify every assistant turn carries `response_routed_experts`.

## What I did

1. Verified the capture proxy from the prior session was still
   listening on port 8001 (`/v1/models` returned 200, sidecar file
   reset to empty).
2. Looped `minimal_agent_driver.py` over the six task IDs:

   | task_id                       | source           | api_calls | wall  |
   |-------------------------------|------------------|----------:|------:|
   | math-tokens-per-gpu-day       | hermes           | 2         | 0.9s  |
   | code-fix-factorial            | hermes           | 3         | 1.3s  |
   | mixed-research-and-math       | hermes           | 4         | 1.6s  |
   | plan-3step-experiment         | hermes           | 2         | 1.4s  |
   | cwc-repo-quality-loop         | cwc_repo         | 2         | 2.2s  |
   | cwc-repo-which-hook-commits   | cwc_repo         | 3         | 1.6s  |

   (Seeded `/tmp/factorial.py` on the spot before the run since the
   `code-fix-factorial` task needs the fixture file.)
3. Pulled `/tmp/minimal_sharegpt_fp8.jsonl` and
   `/tmp/hermes_probes.jsonl` back to the laptop, merged into
   `runs/live/agent/hermes/qwen3-30b-a3b/fp8/<task_id>.json` via
   `scripts/live/merge_probes_into_trajectories.py` with engine
   label `hermes-qwen3-30b-a3b-fp8`.
4. Built the trainer-input payload via
   `scripts/live/build_hermes_dense_input.py --precision-class fp8`,
   which produced `/tmp/rollouts_fp8.json` (16 entries — one per
   probe-bearing assistant turn) and `/tmp/hermes_moe_index_fp8.json`.

## Acceptance evidence

```text
code-fix-factorial.json             assistant_turns=3  with_routed=3  total_gen_tokens=115
cwc-repo-quality-loop.json          assistant_turns=2  with_routed=2  total_gen_tokens=146
cwc-repo-which-hook-commits.json    assistant_turns=3  with_routed=3  total_gen_tokens=117
math-tokens-per-gpu-day.json        assistant_turns=2  with_routed=2  total_gen_tokens=43
mixed-research-and-math.json        assistant_turns=4  with_routed=4  total_gen_tokens=127
plan-3step-experiment.json          assistant_turns=2  with_routed=2  total_gen_tokens=137
```

All 16 assistant turns across 6 trajectories carry non-null
`response_routed_experts` with the expected `[gen_tokens, 48, 8]`
shape. The trainer-input payload `/tmp/rollouts_fp8.json` (also
rsynced back to the spot) is the load-bearing artefact for iters
5+6.

## Notes

* All six tasks completed in `terminal=answered`. None hit
  `max_steps=6` or `max_tokens=1024`.
* `mixed-research-and-math` used the most turns (4 — web_search ×2
  then calculator then answer); `math-tokens-per-gpu-day` and
  `plan-3step-experiment` used the fewest (2 — single tool call
  in parallel or sequence, then answer).
