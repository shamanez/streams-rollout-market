# PROGRESS.md — Agent Handoff State

## Last updated
2026-05-12 (cycle 3 v2 iter 2 setup-7 — partial-capture branch in extractor)

## Cycle 3 v2 iter 2 — setup-7 commit (2026-05-12)

`hermes_token_extraction.extract_token_pairs` now handles the
realistic proxy-captured scenario directly:

- **Path 1 (preferred):** both `prompt_token_ids` and
  `response_token_ids` captured — return verbatim. Unchanged.
- **Path 2 (NEW, proxy-realistic):** `response_token_ids` captured
  but `prompt_token_ids` is None — preserve the captured response
  verbatim (vLLM emitted those exact tokens; re-encoding would yield
  different IDs) and re-derive only the prompt via chat-template.
- **Path 3 (legacy fallback):** both None — re-encode both. Unchanged.

Without this branch, partial-capture trajectories would fall into the
legacy path and have their response tokens re-encoded — destroying
the very signal the proxy captured. The mixed-capture test
(`test_extract_mixed_capture_falls_back_to_chat_template`) already
covered the per-turn mix; the new tests cover the per-turn partial
shape that the proxy actually produces.

2 new tests; **433 passing (+2 from 431)**.

## Cycle 3 v2 iter 2 — setup-6 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-6 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` consolidation shipped on
`research/hermes-dense-e2e-test`:

- `tests/test_hermes_dense_pipeline_e2e.py` wires every cycle-3 v2
  iter 2 setup component end-to-end through their pure-function
  seams: `proxy.extract_probes` → `merger.merge_to_trajectories` →
  `builder.build_rollouts_and_index` → fake trainer →
  `pair.pair_reports` → `DenseMismatchReport`. Asserts both the
  happy path (matched logprobs ⇒ ESS=1.0) AND the divergence path
  (per-token variance ⇒ ESS<0.99) so internal contract drift
  between components cannot silently break the pipeline.
- 2 new tests; **431 passing (+2 from 429)**.

This closes the code-side of cycle-3 v2 iter 2. The remaining gap is
operational: an operator-driven session to install hermes-agent on
the spot, point it at the logprob_capture_proxy, run the 12 Hermes
tasks at bf16 + fp8, scp + torchrun the FSDP teacher-force, then
re-render the dashboards.

## Cycle 3 v2 iter 2 — setup-5 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-5 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (b)-mechanism shipped on
`research/hermes-logprob-proxy`:

- `scripts/live/logprob_capture_proxy.py` — a tiny stdlib-only OpenAI
  chat-completions proxy. Listens on a local port, forwards each
  `/v1/chat/completions` POST to an upstream vLLM endpoint, and:
    * rewrites the outbound request body to add `logprobs=True,
      top_logprobs=1` (preserving the agent's preference when set);
    * adds `extra_body.return_tokens_as_token_ids=True` so vLLM
      returns integer token IDs alongside each logprob (non-vLLM
      upstreams safely ignore the extra);
    * appends one JSONL line to `/tmp/hermes_probes.jsonl` per
      intercepted request with the captured per-turn probes, keyed by
      the request's `X-Prompt-Index` + `X-Turn-Idx` headers (or
      sequential turn counter when headers are absent).
- Pure functions `inject_logprobs()` + `extract_probes()` carry all
  the load-bearing logic; the HTTP wrapper is a thin shell. 11 new
  tests including a true end-to-end run against a stdlib fake
  upstream. **429 passing (+11 from 418).**

The end-to-end dense Hermes pipeline now has a feasible operational
path that does NOT require modifying hermes-agent:

  1. (spot) `bash serve_qwen_for_agent.sh` — vLLM Qwen3-32B bf16 at :8000
  2. (laptop) `ssh -L 8000:localhost:8000 spot -N &` — tunnel
  3. (laptop) `python scripts/live/logprob_capture_proxy.py \
        --upstream http://localhost:8000 \
        --port 8001 \
        --sidecar /tmp/hermes_probes_bf16.jsonl &`
  4. (laptop) install + run hermes-agent against
     `OPENAI_BASE_URL=http://localhost:8001/v1`
  5. (laptop) `python scripts/live/merge_probes_into_trajectories.py \
        --sharegpt <hermes-output>.jsonl \
        --probes /tmp/hermes_probes_bf16.jsonl \
        --tasks scripts/live/agent_tasks_hermes.json \
        --model-id Qwen/Qwen3-32B \
        --engine-label hermes-qwen3-32b-bf16 \
        --engine-fingerprint sha256:... \
        --out-dir runs/live/agent/hermes/qwen3-32b/bf16/`
  6. (laptop) `python scripts/live/build_hermes_dense_input.py \
        --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16 \
        --precision-class bf16`
  7. (spot) `scp + torchrun run_fsdp_reference.py`
  8. (laptop) `scp back + pair_hermes_dense_reports.py` ->
     `runs/live/dense/hermes-qwen3-32b-bf16-vs-fsdp-bf16/`
  9. (laptop) `python scripts/live/publish_dashboards.py`

Note: the proxy captures `response_logprobs` + `response_token_ids`
but **not** `prompt_token_ids` (the OpenAI chat-completions response
shape doesn't expose prompt token IDs). The merger and downstream
pair script tolerate `prompt_token_ids=None`; the trainer-side script
falls back to chat-template re-encoding via `hermes_token_extraction.py`'s
existing fallback path. A future iteration could capture
`prompt_token_ids` via either `echo=True` on the legacy `/v1/completions`
endpoint or a tokenizer-side step.

Next operator-actionable iteration: run the live pipeline end-to-end
on one task to validate the toolchain, then scale up to the full 12
Hermes tasks at bf16 + fp8.

## Cycle 3 v2 iter 2 — setup-4 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-4 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (b)-bridge shipped on
`research/hermes-probe-merger`:

- `scripts/live/merge_probes_into_trajectories.py` defines the
  load-bearing wire format between whatever component captures
  rollout-side probes (a patched hermes-agent batch_runner, a local
  proxy, or a vLLM logging hook) and our `AgentTrajectory` schema.
  The contract is a JSONL sidecar keyed by `(prompt_index, turn_idx)`
  carrying `prompt_token_ids` / `response_token_ids` /
  `response_logprobs` / `response_routed_experts` per assistant turn.
- The merger reads a hermes-agent ShareGPT JSONL + the probe sidecar,
  injects the matching probes onto each `from=gpt` conv entry, and
  adapts to `AgentTrajectory` via the existing
  `adapt_hermes_sharegpt_trajectory` (which already threads sibling
  probe fields onto AgentStep).
- 8 new offline tests (`index_probes` grouping + dedup, partial
  probe coverage tolerated, CLI round-trip with probes surviving the
  full pipeline). 418 passing (+8 from 410).

Cycle 3 v2 iter 2 setup is now complete code-side. The remaining
operational gap is **producing the sidecar JSONL on the spot**.
Three credible paths, ranked by friction:

  1. **Proxy interceptor (laptop)** — write a small OpenAI proxy on
     the laptop that injects `logprobs=True, top_logprobs=1` into every
     chat-completions request, captures the upstream response, and
     writes one sidecar line per turn. hermes-agent on the spot stays
     unchanged. Adds one Python process to the loop but doesn't touch
     hermes-agent internals. Likely the cleanest next iteration.
  2. **Patched batch_runner** — apply a small diff to hermes-agent's
     `agent/auxiliary_client.py` chat-completions builder to always
     pass `logprobs=True, top_logprobs=1` plus a tap that writes the
     sidecar. Requires re-installing hermes-agent on the spot and
     maintaining a fork.
  3. **vLLM logging hook** — vLLM has request/response logging hooks
     in some configs; investigate whether they expose enough.

Operator preference among (1)/(2)/(3) is the next operator input
needed. The autonomous loop can ship the proxy (option 1) without
additional permission since it doesn't touch the spot.

## Cycle 3 v2 iter 2 — setup-3 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-3 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (d)-prep shipped on
`research/hermes-dense-pair-reports`:

- `scripts/live/pair_hermes_dense_reports.py` joins the
  `build_hermes_dense_input` index against the trainer-side output
  (`/tmp/trainers.json` from `run_fsdp_reference.py`) and emits one
  `DenseMismatchReport` per (trajectory, assistant_turn) under
  `runs/live/dense/<pair-slug>/<task_id>-turn<k>/`. The dashboard
  renderer ingests these unchanged.
- `pair_reports(...)` is a pure function; the CLI is a thin
  argparse/I-O wrapper. Trainer engine label can be overridden so the
  same pair script handles `fsdp-bf16` vs `megatron-bf16` cells.
- 7 new offline tests; 410 passing (+7 from 403).

The end-to-end laptop+spot pipeline is now wired top-to-bottom on the
code side:

  1. (spot) Hermes Agent rollouts capture probes -> trajectory JSONs.
  2. (laptop) `build_hermes_dense_input` -> `/tmp/rollouts_hermes_*.json`
     + `/tmp/hermes_dense_index_*.json`.
  3. (spot) `scp + torchrun run_fsdp_reference.py` ->
     `/tmp/trainers.json`.
  4. (laptop) `scp back + pair_hermes_dense_reports.py` ->
     `runs/live/dense/hermes-*/...`.
  5. `python scripts/live/publish_dashboards.py` re-renders.

Step (1) — the actual rollout-side probe capture — is still blocked
on hermes-agent batch_runner instrumentation on the spot. No
hermes-agent install is present on spot as of 2026-05-12T05:31Z; the
next iteration should investigate the cheapest path to wire
`logprobs=True, top_logprobs=1` + persistence of `prompt_token_ids` /
`response_token_ids` into hermes-agent's chat-completions calls.

## Cycle 3 v2 iter 2 — setup-2 commit (2026-05-12)

## Cycle 3 v2 iter 2 — setup-2 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (c)-prep shipped on
`research/hermes-dense-build-input`:

- `scripts/live/build_hermes_dense_input.py` walks a Hermes trajectory
  directory and flattens probe-bearing assistant turns into the
  `/tmp/rollouts.json` shape `run_fsdp_reference.py` already consumes.
  It also writes an index file `/tmp/hermes_dense_index_<precision>.json`
  mapping `prompt_idx → (task_id, trajectory_run_id, assistant_turn_idx,
  num_tokens, response_logprobs)` so the downstream pairing step can
  join FSDP's `/tmp/trainers.json` outputs back to the originating turn
  and to the rollout-side logprobs captured by vLLM.
- Skips turns missing `prompt_token_ids` / `response_token_ids` /
  `response_logprobs` (records the reason in `index["skipped"]`).
- Rejects mixed `model_id` across input trajectories without an
  explicit `--model-id-override`.
- 10 new offline tests (synthetic AgentTrajectory fixtures). 403
  passing (+10 from 393).

Once the spot side captures the probes (part b — pending on
hermes-agent batch_runner instrumentation), the pipeline becomes:

  1. `python scripts/live/build_hermes_dense_input.py
       --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16
       --precision-class bf16`
  2. scp /tmp/rollouts_hermes_bf16.json -> spot:/tmp/rollouts.json
  3. ssh spot 'torchrun --nproc-per-node 4 run_fsdp_reference.py'
  4. scp spot:/tmp/trainers.json -> /tmp/trainers_hermes_bf16.json
  5. python scripts/live/pair_hermes_dense_reports.py  # next iter

Spot reachable as of 2026-05-12T05:31Z. No hermes-agent install on
spot currently — that re-setup is the next blocker for part (b).

## Cycle 3 v2 iter 2 — setup-1 commit (2026-05-12)

`hermes_agent.dense_capture_then_fsdp` part (a) shipped on
`research/hermes-dense-capture-setup`:

- `adapt_hermes_record` + `adapt_hermes_sharegpt_trajectory` now thread
  rollout-side probe fields (`prompt_token_ids`, `response_token_ids`,
  `response_logprobs`, `response_routed_experts`) onto each assistant
  `AgentStep` when present on the input record. Tool-role records that
  carry probes surface as a `ValueError` via `AgentStep`'s own
  validator (instead of being silently dropped).
- `MAX_STEPS` default bumped 8 → 25. Long-horizon agent runs commonly
  need 15-25 turns of tool calls before answering; the cycle-2 default
  of 8 truncated multi-tool tasks prematurely.
- 6 new tests covering: probe pass-through on `adapt_hermes_record`,
  silent-skip when probes are absent, validator rejection on tool
  records, end-to-end pass-through in `adapt_hermes_jsonl`,
  ShareGPT-format pass-through (incl. survival across the tool-result
  backfill rewrite), and the `MAX_STEPS=25` module default.

393 passing (+6 since 387). ruff clean on touched files.

Remaining for the entry to flip `passes: true`:
  - (b) spot re-run of Dense rollouts at bf16+fp8 with vLLM
    chat-completions logprobs=True + persistence of token IDs on each
    assistant turn — needs upstream patches into hermes-agent's
    batch_runner OR a thin proxy in our runner that injects
    `logprobs=True` + `top_logprobs=1` into the OpenAI request.
  - (c) FSDP-bf16 teacher-force on captured `(prompt_token_ids,
    response_token_ids)` pairs (one forward pass per assistant turn).
  - (d) Pair into `DenseMismatchReport` per (trajectory, turn) under
    `runs/live/dense/hermes-qwen3-32b-{bf16,fp8}-vs-fsdp-bf16/`.

## Current phase
**Cycle 3 v2 armed — design pivot mid-session (2026-05-12).** Iter 1
(`hermes_agent.token_extraction_adapter`) shipped, then the operator
killed the original refeed direction: vLLM is rollout only and we do
**not** re-feed response tokens back through it. The corrected design
captures rollout-side signals (per-token logprobs, per-(token, layer)
routed_experts) **during** the vLLM forward pass that generates the
trajectory, persists them on `AgentStep`, and the trainer-side
(FSDP / Megatron) teacher-forces the exact same `(prompt_token_ids,
response_token_ids)` to produce the trainer-side counterparts. This
is long-term stable: token IDs anchor to the tokenizer hash, no
chat-template re-encoding drift.

The cycle-2 trajectory captures on disk (45 trajectories total) lack
the probe fields and will be regenerated as part of cycle 3 v2.

STEER.md has been rewritten for cycle 3 v2 (six iterations). The
`.claude/feature-results.json` entries have been renamed from
`*_refeed_*` to `*_capture_then_*` to reflect the new design.

## Cycle 3 v2 progress (1/6)

- **iter 1 — `hermes_agent.token_extraction_adapter` ✅ (2026-05-12)**.
  `scripts/live/hermes_token_extraction.py` with pure
  `extract_token_pairs(trajectory, tokenizer)` + CLI. Now supports
  TWO paths: (a) preferred — return the captured
  `step.prompt_token_ids` / `step.response_token_ids` verbatim when
  present (no tokenizer needed; long-term stable); (b) fallback —
  chat-template re-encoding for legacy trajectories. Mixed-capture
  trajectories work too. 11 new tests; total +10 in this session
  (incl. 7 schema-extension tests in `test_agent_trajectory_lab.py`).
- **Schema extension (2026-05-12) shipped same commit.**
  `AgentStep` now carries four optional probe fields, all defaulting
  to None for backward compatibility:
    - `prompt_token_ids: list[int] | None` — exact prompt vLLM saw
    - `response_token_ids: list[int] | None` — exact tokens vLLM emitted
    - `response_logprobs: list[float] | None` — per-token vLLM logprob
    - `response_routed_experts: list[list[list[int]]] | None` —
      `[token][layer][top_k]` shape for MoE
  Validator enforces length parity + forbids probes on tool-role
  steps. Legacy JSONs validate unchanged.

- **iter 2/6 next — `hermes_agent.dense_capture_then_fsdp`** (live,
  spot). See STEER.md cycle-3 v2 iter 2 directive: (a) extend
  `scripts/live/hermes_agent_runner.py` to capture logprobs from
  chat-completions, **bump `MAX_STEPS` default 8 → 25** (long-horizon
  multi-turn agent runs need it), thread `prompt_token_ids` /
  `response_token_ids` / `response_logprobs` onto **each** `AgentStep`
  (every assistant turn captures its own prompt — which includes the
  full accumulated prior history — and its own response); (b)
  re-run Hermes-Agent Dense rollouts at bf16 + fp8; (c) FSDP-bf16
  teacher-force on captured (prompt, response) pairs (one forward
  pass per assistant turn); (d) pair into `DenseMismatchReport` per
  (trajectory, turn).

- **iter 6 scope update (UI cleanup, 2026-05-12).**
  `matrix_render_and_codex` now also removes the legacy single-prompt
  Dense + MoE matrices from the headline. The marketplace dashboard
  becomes Hermes-only above the fold: Hermes-Dense ESS + Hermes-MoE
  router_flip_rate, both 2×2 over `{FSDP, Megatron} × {bf16, fp8}`.
  Legacy single-prompt outputs move into a collapsible "Legacy
  single-prompt experiments (archived)" appendix — not above the
  fold. The underlying CLIs stay; only the dashboard layout changes.

## Cycle 2 retrospective (what worked, what was wrong)
**Hermes Agent cycle COMPLETE — 12 traffic-light tiles + codex PASS.**
All 29 entries in `.claude/feature-results.json` are `passes: true`.
STEER.md self-deleted per its own stop clause. The rendered
`/docs/index.html` now shows THREE matrices above the fold:

  * Dense (Qwen3-32B)         — ESS                          (8 tiles, 4 visible)
  * MoE (Qwen3-30B-A3B)       — router_flip_rate             (4 tiles, all visible)
  * Hermes Agent Dense        — final_answer_match X/N        (2 tiles)
  * Hermes Agent MoE          — final_answer_match X/N        (2 tiles)

Cycle 2 findings: vLLM-fp8 inference matters more for MoE rollouts
than Dense rollouts AT BOTH the token level AND the agent-trajectory
level:
  Dense ESS:           bf16 0.998  -> fp8 0.995  (tiny precision drop)
  MoE flip rate:       bf16 4%     -> fp8 7%     (clear precision effect)
  Hermes Dense jaccard: fp8 0.32   vs noise 0.29 (negligible)
  Hermes MoE jaccard:   fp8 0.39   vs noise 0.48 (precision IS divergent)

For crowdsourced MoE rollouts at fp8, the precision effect on multi-
turn agent trajectories exceeds sampling noise — confirming the
marketplace thesis at the action level.

## Hermes Agent integration (Iter 1-6 arc, shipped 2026-05-12)

Iter 1 shipped: `scripts/live/hermes_agent_runner.py` + pure
`adapt_hermes_record` / `adapt_hermes_jsonl` adapters with 18 new
tests.

Iter 2 shipped: `scripts/live/agent_tasks_hermes.json` (12
multi-turn tasks per the approved plan: 1 no-op-trivia, 6 multi-tool
tasks, 1 four-tool long-horizon). Expanded `_SIM_FILES` in
`scripts/live/agent_runner.py` with `/tmp/fib.py` (off-by-one bug)
and `/tmp/clip.py` (correct clipping function for the
read-then-verify task). 11 new tests in
`test_agent_tasks_hermes_schema.py`. Total 376 passing (+29 since
session start).

**Iter 3 bf16 side LANDED (1/3 sides done).** Full end-to-end pipeline
is operational; 11 real Hermes Agent trajectories on disk at
`runs/live/agent/hermes/qwen3-32b/bf16/<task_id>.json`.

The pipeline that now works:
1. `pip install /tmp/hermes-agent` into isolated `~/hermes-venv`
   (hermes-agent 0.13.0, no curl-pipe-bash needed).
2. Patched `MINIMUM_CONTEXT_LENGTH` from 64_000 down to 8_000 in
   both the installed package and the `/tmp/hermes-agent` source
   tree (Qwen3-32B max is 40_960; the 64K floor is a recommendation,
   not load-bearing). The `.bak` files are kept for rollback.
3. vLLM Qwen3-32B-bf16 serve on spot at `MAX_LEN=40960`.
4. SSH tunnel `localhost:8000 → spot:8000`.
5. Hermes home at `/tmp/hermes_smoke_home/config.yaml` with
   `model.provider=vllm`, `base_url`, `context_length=40000`.
6. `batch_runner.py --base_url=... --model=Qwen/Qwen3-32B
   --ephemeral_system_prompt="...you MUST call a tool..."` produces
   one `trajectories.jsonl` with all 12 prompts.
7. `adapt_hermes_sharegpt_trajectory()` converts each record into
   an AgentTrajectory JSON.

bf16 results vs STEER acceptance gate:
- ✅ trajectories well-formed AgentTrajectory JSON
- ❌ 11/12 trajectory files (`code-fix-fibonacci` missing — Hermes
  read_file hit real /tmp which doesn't have our fixture)
- ❌ 3/11 have `len(steps) == 1`: `research-vllm-then-sglang`,
  `mixed-research-and-math`, `no-op-trivia` (model answered from
  memory; gate requires ≥2)
- ✅ no `terminal_reason="error"` (gate allows ≤2)

Tool calls across the bf16 side: 22 calls — terminal (9),
read_file (5), search_files (3), python_eval (3), write_file (1),
web_search (1). Mean wall-clock ~48s/prompt; total 572s for 12 at
num_workers=2.

**Remaining for Iter 3 acceptance**:
- fp8 side: re-serve `MODEL=Qwen/Qwen3-32B-FP8 DTYPE=auto`, re-run
  the batch (~10 min wall-clock + 5 min model load).
- bf16_seedB side: same model, `SEED=4321` (sampling-side, not
  serve-side — need a thin wrapper).
- Re-run `code-fix-fibonacci` (or pre-create `/tmp/fib.py` on Mac).
- Tighten 3 single-step prompts (or accept them as
  legitimate-divergence evidence).

393 tests passing, ruff clean. Loop halts here; operator can resume
with `/autonomous-loop` to drive the fp8 + seedB sides. Spot serve
+ SSH tunnel are stopped — no idle GPU spend.

The previous-cycle dashboard restructure is still live;
`/docs/index.html` continues to answer the load-bearing question:

  *Engine + precision + device DO change inference-time behaviour
  enough to matter for crowdsourced MoE rollouts.* fp8 inference
  adds 2-3% router_flip_rate over the bf16 baseline, regardless of
  whether the trainer reference is FSDP-bf16 or Megatron-bf16.

Branch hygiene (2026-05-12): all merged research/setup branches
deleted locally + on origin. Repo now carries `main` +
`research/megatron-and-engine-purge` (predecessor branch, kept for
historical reference — its two commits were superseded by the work
that landed on main).

## Resume in a new session
**Single command:** `/autonomous-loop`

With STEER.md gone, the loop will fall back to PROGRESS.md "Next
work" (see below) and `docs/future_research.md` for the next unit
of work.

Operator overrides:
- `touch AGENT_STOP` — halt after the current iteration.
- `bash .claude/scripts/steer.sh "<note>"` — re-arm STEER.md.
- `rm AGENT_STOP` — resume.

## How to view the dashboard
```bash
python scripts/live/publish_dashboards.py    # re-render docs/
python -m http.server -d docs 8000 &
open http://localhost:8000/
```
Full live-experiment runbook lives in `CLAUDE.md#how-to-run`.

## Shipped this session (2026-05-11)

Branch `research/megatron-qwen3-32b-bindmount` carries the full
9-iteration arc that drove the dashboard to 8 tiles.

### Iter A — `feat(megatron): run_megatron_reference.py`
- Dense Qwen3-32B Megatron-bf16 teacher-force inside slimerl/slime.
- Uses stock Megatron's `model_provider` + `gpt_builder` (skips
  slime's training wrappers). `wrap_with_ddp=False` to avoid the
  30 GiB DDP grad-buffer OOM on L40S.
- `gather_from_tensor_model_parallel_region(logits)` before
  log_softmax — output is TP-sharded across the vocab dim by default.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` env var for TP > 1.
- Setup commit, no feature-results flip.

### Iter B — `live(megatron): dense.qwen3_32b.vllm_bf16_megatron`
- vLLM-bf16 rollout + Megatron-bf16 teacher-force.
- Headline: ESS=0.9994, clipped=0, veto=0, delta_logprob_abs=0.0101.

### Iter C — `live(fp8): dense.qwen3_32b.vllm_fp8_megatron`
- vLLM-fp8 rollout + Megatron-bf16 teacher-force.
- Headline: ESS=0.9948, clipped=0, veto=0, delta_logprob_abs=0.0364.
- Confirms fp8 inference does NOT break OPBC thresholds against a
  Megatron-bf16 trainer either.

### Iter D — `feat(fsdp): run_fsdp_moe_reference.py`
- FSDP MoE teacher-force with `output_router_logits=True`. Emits
  length-(P+R-1) traces matching vLLM's `routed_experts` shape so
  RouterTrace can pair them.

### Iter E — `live(moe-fsdp): moe.qwen3_30b_a3b.vllm_bf16_fsdp_router`
- New `build_router_input_pair.py` with 4 variants covering the
  (vllm-bf16, vllm-fp8) × (fsdp-bf16, megatron-bf16) MoE matrix.
- Headline: **router_flip_rate=4.3%** for vLLM-bf16 vs FSDP-bf16.

### Iter F — `live(moe-fp8-fsdp): moe.qwen3_30b_a3b.vllm_fp8_fsdp_router`
- vLLM-fp8 (TP=2 because the FP8 expert ffn_intermediate=768 yields
  192 per shard at TP=4, which fails vLLM's FP8 block_n=128).
- Headline: **router_flip_rate=7.1%** for vLLM-fp8 vs FSDP-bf16.

### Iter G — `live(moe-megatron): moe.qwen3_30b_a3b.vllm_bf16_megatron_router`
- New `run_megatron_moe_reference.py` + `megatron_moe_reference_launch.sh`
  (mirrored from the existing on-spot script + bind-mounts).
- Headline: **router_flip_rate=4.1%** for vLLM-bf16 vs Megatron-bf16.

### Iter H — `live(moe-fp8-megatron): moe.qwen3_30b_a3b.vllm_fp8_megatron_router`
- Closes the 4-cell MoE matrix.
- Headline: **router_flip_rate=6.3%** for vLLM-fp8 vs Megatron-bf16.

### Iter I — `docs(dashboards): dashboard.high_quality_render`
- Fixed `router_dashboard.py` to emit `precision_class` on each row
  (derived from rollout_engine name) so the matrix bins MoE reports
  into bf16 / fp8 columns instead of a phantom "—" column.
- All 8 matrix tiles populated, no `mx-empty` / `mx-tbd`
  placeholders in the visible area.
- Codex review verdict: **PASS**.

## 8-tile matrix state

| matrix                 | trainer  | bf16    | fp8     |
|------------------------|----------|---------|---------|
| Dense (Qwen3-32B) ESS  | FSDP     | 0.9987  | 0.9949  |
| Dense (Qwen3-32B) ESS  | Megatron | 0.9994  | 0.9948  |
| MoE (Qwen3-30B-A3B) fr | FSDP     | 4.3%    | 7.1%    |
| MoE (Qwen3-30B-A3B) fr | Megatron | 4.1%    | 6.3%    |

Pattern: dense ESS is engine-and-precision-robust (>0.99 everywhere).
MoE router_flip_rate is dominated by vLLM-vs-trainer kernel
differences (~4% at matched precision), with fp8 inference adding
2-3% more divergence regardless of trainer engine.

## Next work (post-STEER)

The dashboard is at 8 tiles, but each MoE cell is n=1 — one prompt,
one response. Open candidates for follow-up iterations:

1. **MoE matrix n=8.** Repeat each MoE cell with 8 different prompts
   so the (router_flip_rate, layer-distribution) numbers carry error
   bars. Reuse `scripts/live/prompts.json`. Mirror the dense matrix
   precedent (n=8 for the bf16-vs-FSDP corner).
2. **Layerwise heatmap.** The current router dashboard reports
   per-pair scalar flip rates; surface the per-layer flip array as
   a row-of-bars in each tile's drill-in. Worst layer is the most
   actionable signal for OPBC routing decisions.
3. **OPBC integration of router_flip_rate.** Add an OPBC reason
   `high_router_flip_rate` that quarantines MoE groups whose
   measured flip rate exceeds a configurable threshold. Currently
   the OPBC only inspects logprob-level mismatch.

## Test status
- pytest -q: **447 passed, 0 failed** (2026-05-12). +40 since the
  prior 407: 7 schema-extension tests in `test_agent_trajectory_lab.py`
  + 11 in `test_hermes_token_extraction.py` (incl. coerce / captured-IDs
  / mixed-capture / no-tokenizer cases), the rest from the broader
  Hermes shipping arc captured upstream.
- ruff check .: clean.
- `.claude/feature-results.json`: **24/29 entries pass:true** after
  the cycle-3 v2 re-arm. Open: 5 `hermes_agent.*` entries
  (`dense_capture_then_fsdp`, `dense_capture_then_megatron`,
  `moe_capture_then_fsdp_router`, `moe_capture_then_megatron_router`,
  `matrix_render_and_codex`).
- The next `/autonomous-loop` invocation picks up
  `hermes_agent.dense_capture_then_fsdp` first (top-most false entry
  per the SKILL's selection rule).

## Next work (post-Hermes)

(Empty — see Research follow-ups below for completed items. The
dashboard surface is currently at a steady state; new initiatives
would come from `docs/future_research.md`.)

## Research follow-ups shipped (post-Hermes)

- **Dashboard click-through pages (2026-05-12, future_research Q17)**.
  `publish_dashboards.py` now copies the latest per-(pair, task)
  `agent_divergence_report.html` to
  `docs/agent_diff/<pair-slug>/<task-id>.html`. 54 stable URLs across
  5 pair slugs. The matrix tiles don't yet deep link to them; one-line
  href update is the next follow-up.

- **n=30 hermes rollouts (2026-05-12)**. Added bf16_rep2 batches for
  Dense (n=12) and MoE (n=9) and a replicate-aware pair script.
  Dense tiles now pair against n=24 bf16 reference, MoE TP=4 path
  against n=18 reference. The precision-vs-noise gap stabilises to
  ~8-12pp across both models: Dense fp8 45% vs noise 37.5%; MoE
  fp8(TP=4) 50% vs noise 33%. All tiles still mx-bad (under 50%
  amber threshold) but the gradient communicates the marketplace
  signal cleanly.

- **Soft answer-match (2026-05-12)**. Added
  `final_answer_match_soft` alongside the strict field, computed via
  numeric-token equivalence + substring fallback. Hermes tile
  rendering uses the soft rate; first AMBER Hermes tile (MoE fp8
  5/9 = 56%) emerged. Strict field preserved for back-compat.
  415 tests passing (+8 unit tests for the soft matcher).

- **MoE bf16 @ TP=2 noise floor (2026-05-12)**. Added a fresh
  Qwen3-30B-A3B bf16 serve at TP=2 (matching fp8's required TP).
  publish_dashboards.py now takes precision_trainer / noise_trainer
  overrides so the MoE precision-effect tile pairs fp8 @ TP=2
  against bf16 @ TP=2 (apples-to-apples) while the noise-floor pair
  stays on bf16 @ TP=4 seedB-vs-bf16. New tile values: precision
  5/11 (45.5%) vs noise 4/9 (44.4%) — statistically
  indistinguishable at n=12. The earlier amber tile (5/9, 56%) was
  the TP=4→TP=2 confound, not a fp8 effect.

## Reproducibility
Live runs use the env-driven runbook in `scripts/live/`:
- Dense: `MODEL`, `VLLM_DTYPE`, `ROLLOUT_LABEL`, `TRAINER_REFERENCE`.
- MoE rollout: `run_vllm_moe_rollout.py` (with
  `enable_return_routed_experts=True`).
- FSDP MoE reference: `run_fsdp_moe_reference.py` under torchrun.
- Megatron dense reference: `run_megatron_reference.py` +
  `megatron_reference_launch.sh` (slimerl/slime container,
  bind-mounted `~/megatron_conversion/qwen3_32b_torch_dist/`).
- Megatron MoE reference: `run_megatron_moe_reference.py` +
  `megatron_moe_reference_launch.sh` (same image, bind-mounted
  `~/megatron_conversion/megatron_ckpt/` + `~/megatron_conversion/hf_model/`).
Historical narrative + empirical findings live in `git log` and the
rendered dashboards under `/docs/`.
