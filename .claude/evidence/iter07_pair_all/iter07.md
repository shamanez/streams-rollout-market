# Iter 07 (cycle FP8) — pair all 4 (rollout × trainer) combos

## Directive

Run `pair_hermes_moe_reports.py` four times, one per (rollout, trainer)
pair, producing 4 cells × N trajectories of `router_mismatch_report.json`.

## What I did

Wiped the prior `runs/live/router/hermes-qwen3-30b-a3b-*` reports
(those were the n=1 single-trajectory cells from cycle-3-v2), then
ran the pair script four times. Each run pulls per-trajectory data
from `runs/live/agent/hermes/qwen3-30b-a3b/<prec>/` (whose
`rollout_engine` field is `hermes-qwen3-30b-a3b-{bf16,fp8}`) and
joins it against the trainer-side router trace from iter 5/6.

```
python scripts/live/pair_hermes_moe_reports.py \
    --index /tmp/hermes_moe_index_<prec>.json \
    --trainers <trainer_router_json> \
    --trajectory-dir runs/live/agent/hermes/qwen3-30b-a3b/<prec>/ \
    --trainer-engine-label <fsdp|megatron>-bf16-moe \
    --out-root runs/live/router/hermes-qwen3-30b-a3b-<prec>-vs-<trainer>-bf16-moe
```

70 reports written total (19 bf16 + 19 bf16 + 16 fp8 + 16 fp8).

## Acceptance evidence — the 2×2 matrix

|                     | FSDP-bf16 (trainer)             | Megatron-bf16 (trainer)         |
|---------------------|--------------------------------:|--------------------------------:|
| **vLLM bf16 rollout** | **6.01%** mean (n=19, 3.00–9.63%) | **5.72%** mean (n=19, 3.03–9.12%) |
| **vLLM fp8 rollout**  | **8.29%** mean (n=16, 5.18–12.76%) | **8.34%** mean (n=16, 4.95–12.85%) |

Per CLAUDE.md bands: green ≤ 5%, amber 5–15%, red > 15%. All four
cells are **amber**, both bf16 cells skating close to the green
boundary, both FP8 cells decisively inside amber.

## What this says

1. **FP8 quantization adds ~2.3-2.6 percentage points** to the
   rollout-vs-trainer router_flip_rate. The trainer is always bf16,
   so this is the marketplace cost of accepting FP8-quantized
   rollouts into a bf16 GRPO trainer.
2. **The trainer engine barely matters**: FSDP and Megatron agree
   within 0.3 pp on every cell. FSDP runs HF transformers with
   `output_router_logits=True`; Megatron uses its own `mlp.router`
   forward hook. Different code paths, near-identical numbers — a
   strong consistency signal for the metric itself.
3. **Range matters as much as mean**. The bf16 cells have flip
   rates as low as 3.0%, but the FP8 cells bottom out at 5%. The
   FP8 quantization narrows the "easy tokens" floor — even tokens
   that bf16 routes deterministically (≤3% across rollouts ↔
   trainer) get jostled in FP8.

## Note on rollout_engine labelling

The rollout engine name flows from each trajectory JSON
(`merge_probes_into_trajectories.py --engine-label` set it at iter
3+4), not from a pair-script CLI arg. So all 19 bf16 trajectories
land under the bf16 rollout column, and the 16 FP8 ones under the
FP8 column, automatically.
