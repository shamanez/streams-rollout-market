# Future research and open problems

A running, opinionated list. Each item lists what we don't yet know,
why it matters, and roughly what it would take to get an answer with
the existing rig.

## Open empirical questions

### 1. Does engine drift translate into a *training-step* difference?
We measure ESS, |Δlogp|, top-k set disagreement, and tool-call
divergence — all proxies. The load-bearing claim ("rollouts that look
fine token-by-token become a different agent at the trajectory level")
is supported, but we have not yet shown that an actual GRPO or PPO
update under engine A reaches a different parameter state than the
same update under engine B with the same group. Need a small,
controlled training run that consumes ~100 groups produced by both
engines and compares the resulting parameter delta. Would take a
day of spot work plus a tiny GRPO loop (out of project scope but
*usable from* the project's contracts).

### 2. Statistical significance on single-prompt findings
Many of the headline numbers are n=1 (the MoE router) or n=8 (the
dense + agent matrices). The directionality is robust, but the
absolute numbers (e.g. "FP8 changes the final answer on 67% of
tasks") need n ≥ 30 to mean anything tighter than ±15% confidence.
~3 hours of spot work to run the existing matrix at n=30.

### 3. Sequence-length scaling
We've only run 128-token responses for the dense matrix and 64 tokens
for the MoE router. The interesting question is how ESS evolves as
the response grows: linearly (drift accumulates per token) or
super-linearly (clipping kicks in past some threshold). Would inform
*where* the "trainable" cutoff sits as RL training drifts toward
longer-horizon agentic tasks. Easy follow-up: same matrix at 128 /
512 / 2048 token responses.

### 4. Cross-hardware: H100, H200, MI300X
All current data is from L40S — a Lovelace-class card without native
FP8 matmul. On Hopper (H100/H200), FP8 has hardware support, so the
quantization-induced drift may *narrow* significantly. On AMD
(MI300X), the kernel stack is different end-to-end and we expect new
drift modes. Either flips or strengthens the FP8 finding. Each new
hardware family is ~4 hours of ops setup the first time.

### 5. Megatron-LM as a trainer-side reference
FSDP forward-only is bit-equivalent to HF transformers (we measured).
Megatron uses different attention and MLP CUDA kernels even in
forward, so it's the next trainer engine that would actually move the
numbers. Blocker: HF→Megatron checkpoint conversion. ~1 day session.
The end-to-end recipe lives in
[`scripts/live/megatron_convert_qwen3_moe.md`](../scripts/live/megatron_convert_qwen3_moe.md)
— a pinned `slimerl/slime:latest` docker run that builds a
Qwen3-30B-A3B torch-dist checkpoint on the spot instance (~200 GB
peak, ~80 min total). Once the dist-ckpt lands, the live scripts pick
it up via `TRAINER_REFERENCE=megatron`; no code changes needed.

### 6. MoE router under longer horizons and multi-prompt
The 98.4% top-k set disagreement is from one 64-token sample. Need
to (a) extend to multi-prompt, (b) check whether the disagreement
*concentrates* at certain layer positions (early/late) when n is
larger, (c) see how this scales with the number of routed experts.
The contract layer is ready; just need to run more.

### 7. Real (vs simulated) tools in the agent matrix
The 6 tasks use deterministic stub tools (`web_search` returns
canned text). That's deliberate — it isolates the engine effect from
tool-output variance. But the next step is real tools (actual web
search via SerpAPI free tier, real `python_eval`) so the
trajectory-level findings transfer to "what happens in production."
Roughly a day of careful sandboxing.

### 8. Multi-turn / multi-thousand-step agents
Current trajectories are 5-8 steps. The interesting question is
whether engine drift compounds linearly, sub-linearly, or
super-linearly over hundred-step research-agent runs. This is where
the "compounding mismatch" thesis would be either definitively shown
or partially refuted. Needs a longer-horizon task suite.

### 9. Token-level mismatch *within* agent trajectories
We measure trajectory-level (tool choice, answer match) and
token-level (dense lab) separately. Tying them together — "the agent
diverged at step 3 because token 47 within step 2's response had a
log_ratio of 0.95" — is a single-step downstream of the existing
code. Would let the agent dashboard click through into the dense
view of the assistant turn that caused the divergence.

## Open design questions

### 10. What is the right "trainable" threshold?
OPBC's defaults — `min_ess=0.30`, `max_clipped_fraction=0.10`,
`max_policy_lag_steps=8` — are reasonable but arbitrary. Empirical
study: under what threshold combinations does a real GRPO step still
converge? Without that, the marketplace's `train` /
`train_with_correction` / `replay` / `quarantine` boundary is a
guess.

### 11. Engine fingerprint as a contract field
Today `precision_class` (bf16 / fp8) pins precision and
`engine_contract_version` pins the API surface, but **the actual
kernel implementation** isn't part of the policy manifest. Two vLLM
0.20.1 builds with different attention backends produce different
logprobs. Should the `WorkerManifest` carry a numerical fingerprint
(e.g., output of a canonical-input forward pass) the dispatcher can
check?

### 12. Trainer-feedback granularity
The `FeedbackAggregator` rolls up by `(worker_id, engine_name,
policy_version)`. If `engine_name` is too coarse (vLLM but not which
build) or too fine (every engine version), the dispatcher's
quality-bias signal becomes noisy. The right granularity probably
involves the engine numerical fingerprint above.

### 13. Cross-process worker quality signal
`FeedbackAggregator` is in-process today. In production, trainer
feedback would arrive from a different process (or different host)
than the dispatcher. The contract is fine; the storage layer needs
swapping. Open: pull-based vs push-based, retention window, eviction
policy.

### 14. Quorum reload in production
`PoolReloadTracker` is a nice unit-tested abstraction; it has not
been run live. Open: do real workers reliably ack a checkpoint
after weights are loaded? What happens when ack arrives before the
worker has dropped its old KV cache? The contract layer assumes ack
implies "ready to serve vN" — we should validate this with a live
multi-worker reload demo.

### 15. Replay-tier semantics under multi-trainer pull
`TrainerClient` increments `served_count` on the LiveStore record.
With multiple trainers pulling concurrently, a popular replay-tier
group could be served many times. Open: should `served_count` carry
a per-trainer breakdown, or is global enough? Affects how a
dispatcher would price replay vs fresh.

## Operational follow-ups

### 16. `runs/<UTC-ts>` UUID suffix landed but…
…the *fixture* directories still live under `/tmp/fixtures/<engine>`
which is wiped on every script run. Move them under `runs/live/`
(gitignored) so a session-spanning history is preserved.

### 17. ~~Dashboard "click through to detail"~~ **(shipped 2026-05-12, commit d27bf29)**
`scripts/live/publish_dashboards.py::_copy_agent_diff_click_through`
walks `runs/live/agent_diff/<pair-slug>/<ts-uuid>/agent_divergence_report.{json,html}`,
picks the latest replicate per `(pair_slug, task_id)`, and emits
`docs/agent_diff/<pair-slug>/<task-id>.html`. 54 pages currently live
under `docs/agent_diff/` (5 pair slugs × ~11 tasks). The matrix tiles
don't yet deep-link to these pages — that's a one-line href change
on the next iteration. Follow-up: surface a per-pair task-list table
in the agent dashboard with direct links.

### 18. `serve_on_spot.sh` should detect IP rotation
The spot's public IP is dynamic across stop/start. The script
re-reads it each run; what it doesn't do is *notice when the cached
IP from the last run is stale*. A small "if you set up a tunnel, the
URL has changed" warning would help.

### 19. Multi-region / distributed worker demo
`LocalWorkerBroker` is in-process and thread-safe. The contract is
ready for a multi-process broker (Postgres / Redis / NATS) but no
such backend exists yet. The next deployment story is the demo of
one trainer in one region pulling from workers in two AWS regions.

## Research questions that would change the project

### 20. Is the contract layer sufficient, or do we need *attestation*?
Today a worker self-reports `engine_name`, `engine_version`,
`precision_class`. A malicious worker could lie. For high-stakes
training (real money on the line) we'd need cryptographic
attestation — the worker proves the rollout came from a specific
binary on specific hardware. Out of scope for v0; relevant if the
marketplace ever accepts external workers.

### 21. Is the OPBC's set of decision reasons exhaustive?
Today: WITHIN_BUDGET, HIGH_CLIPPED_FRACTION, LOW_ESS,
STALE_POLICY_LAG, REPLAY_TIER_LAG, REPLAY_TIER_ESS,
VETO_THRESHOLD_EXCEEDED. We've not yet seen a real rollout we
couldn't classify. Open: is there a class of pathology that doesn't
fit (e.g., adversarial prompts that produce contract-clean but
semantically-broken outputs)?

### 22. The marketplace economics
None of the current code prices anything. Worth a separate document:
how does a worker get paid for a `train`-classified group vs a
`replay`-classified one? Does the trainer post a bounty? How does
quality-of-service affect future bid acceptance? This is the layer
above the contract that turns it into a market.
