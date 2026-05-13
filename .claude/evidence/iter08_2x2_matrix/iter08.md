# Iter 08 (cycle FP8) — aggregate + republish 2×2 router matrix

## Directive

Re-aggregate `router_dashboard.json`, update the headline matrix to
show one column per rollout precision, republish. Acceptance: 2×2
matrix with 4 traffic-light tiles, each `n ≥ 3`.

## What I did

1. **Re-aggregated.** `python -m rollout_market.cli.router_dashboard
   --reports-glob 'runs/live/router/**/router_mismatch_report.json'`
   collected all 70 reports from iter 7 into four `engine_pairs`
   entries keyed on `(rollout_engine, trainer_engine)`:

   ```
   bf16 × FSDP:     mean_flip 0.0601 (n=19)
   bf16 × Megatron: mean_flip 0.0572 (n=19)
   fp8 × FSDP:      mean_flip 0.0829 (n=16)
   fp8 × Megatron:  mean_flip 0.0834 (n=16)
   ```

2. **Extended the renderer** (`_render_hermes_router_matrix` in
   `scripts/live/publish_dashboards.py`): replaced the single
   `rollout_engine: str` arg with `rollout_engines: tuple[str, ...]`
   and emit one `<div class='mx-col-head'>` per rollout. The two
   trainer rows now each carry one cell per rollout precision.
   Also added an optional `column_labels` arg so the headline reads
   `vLLM bf16 · L40S` / `vLLM fp8 · L40S` instead of the raw engine
   slug.

3. **Republished** via `publish_dashboards.py`. Resulting matrix:

   ```
   data-section="hermes-agent-moe-router-matrix"
     mx-col-head count: 2       (vLLM bf16, vLLM fp8)
     mx-row count: 2            (FSDP, MEGATRON)
     mx-tile count: 4           (one per cell)
     tile values: 6.01% / 8.29% / 5.72% / 8.34%   (all amber)
     per-tile n: 19 / 16 / 19 / 16
   ```

## Acceptance evidence

`docs/index.html`'s router matrix section now contains:

```html
<section data-section="hermes-agent-moe-router-matrix">
  <h2>Hermes Agent — MoE router (Qwen3-30B-A3B)</h2>
  <div class='mx-grid'>
    <div class='mx-grid-head'>
      <div>trainer-ref</div>
      <div>vLLM bf16 · L40S</div>
      <div>vLLM fp8 · L40S</div>
    </div>
    <div class='mx-row'>
      <div>FSDP</div>
      <a class="mx-tile mx-warn">… 6.01% … n=19 run(s) · 703 tok · worst layer 33.3% …</a>
      <a class="mx-tile mx-warn">… 8.29% … n=16 run(s) · 685 tok · worst layer 41.7% …</a>
    </div>
    <div class='mx-row'>
      <div>MEGATRON</div>
      <a class="mx-tile mx-warn">… 5.72% … n=19 run(s) · 703 tok · worst layer 37.5% …</a>
      <a class="mx-tile mx-warn">… 8.34% … n=16 run(s) · 685 tok · worst layer 41.7% …</a>
    </div>
  </div>
</section>
```

* 4 tiles ✓
* 2 columns (bf16, fp8) × 2 rows (FSDP, Megatron) ✓
* Per-tile `n=19` (bf16) or `n=16` (fp8) — well above the `n ≥ 3`
  acceptance threshold ✓
* Tests: 471 passed; ruff: clean ✓

## Story the matrix tells

* **Trainer engine choice doesn't matter** at this scope: FSDP and
  Megatron agree to within 0.3 pp on every cell.
* **Rollout precision matters a lot**: FP8 quantization adds
  ~2.3-2.6 pp of router_flip_rate vs bf16 rollouts (6% → 8.3%).
  All cells stay inside the amber band, but FP8 is decisively
  worse.
* **Worst-layer flip rate** rises from ~37% (bf16) to ~42% (FP8) —
  the hardest single layer also gets harder under FP8.
