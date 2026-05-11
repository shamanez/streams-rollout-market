"""Plain-English explanations of every dashboard metric.

STEER ``dashboard.glossary_verl_rewrite`` mandates a four-field shape
for every glossary entry, paraphrasing the language verl uses for its
train-inference correction module:

1. **what** — what the metric is, in one line.
2. **aggregation** — per-token, per-sequence, or per-trajectory.
3. **cap** — the threshold beyond which OPBC takes action, named
   alongside the constant that holds it (e.g. ``BudgetPolicy.clamp =
   20.0 nats``). Capability and observability metrics that don't have a
   cap state ``no cap — observability only`` explicitly so readers
   don't go searching.
4. **drop_behavior** — what happens when the cap fires: the token is
   clipped in place, or the entire group is quarantined, or no drop
   (observability only).

A longer ``long`` field carries extra prose context. It is optional and
purely cosmetic — the test suite asserts only the four required fields.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryEntry:
    """One glossary metric, formatted in the verl four-field shape."""

    what: str
    aggregation: str
    cap: str
    drop_behavior: str
    long: str = ""


# The four-field shape applies to every entry. Constants referenced in
# the ``cap`` field live in:
#   * ``rollout_market.opbc.BudgetPolicy``
#       clamp = 20.0 (nats), veto_abs_log_ratio = 30.0 (nats),
#       max_clipped_fraction = 0.10, min_ess = 0.30,
#       replay_ess_threshold = 0.60.
#   * ``rollout_market.mismatch_metrics.summarize_logprob_mismatch``
#       (same clamp / veto_abs_log_ratio defaults).
GLOSSARY: dict[str, GlossaryEntry] = {
    "ESS": GlossaryEntry(
        what=(
            "Effective sample size of importance weights — how usable a "
            "rollout is for off-policy training."
        ),
        aggregation="per-sequence (one ESS value per group / response).",
        cap=(
            "soft floors at `BudgetPolicy.replay_ess_threshold = 0.60` and "
            "`BudgetPolicy.min_ess = 0.30`; no per-token clamp."
        ),
        drop_behavior=(
            "below 0.60 routes the group to `replay`; below 0.30 routes it "
            "to `quarantine`. The tokens themselves are not modified."
        ),
        long=(
            "ESS = (Σw)² / (N · Σw²) where w is the importance weight per "
            "token. ESS=1 means the rollout matches the trainer's policy "
            "exactly. ESS dropping toward 0 means the rollout is "
            "increasingly off-policy and a trainer would need stronger "
            "correction (or skip the rollout)."
        ),
    ),
    "|Δlogp|": GlossaryEntry(
        what=(
            "Per-token disagreement size between rollout and trainer "
            "logprobs, in nats."
        ),
        aggregation=(
            "per-token; dashboards display the mean of |Δlogp| over the "
            "response."
        ),
        cap=(
            "`BudgetPolicy.clamp = 20.0 nats` (defined in "
            "`rollout_market.opbc.BudgetPolicy`); shared with "
            "`mismatch_metrics.summarize_logprob_mismatch(clamp=20.0)`."
        ),
        drop_behavior=(
            "tokens with |Δlogp| > clamp are clipped in place to ±20 nats; "
            "the rest of the group survives."
        ),
        long=(
            "Mean over the response of |trainer_logp(token) − "
            "rollout_logp(token)|. Tiny values (~0.01) mean the two engines "
            "agree on most tokens; values > 0.1 mean meaningful single-"
            "token drift."
        ),
    ),
    "log_ratio": GlossaryEntry(
        what=(
            "log( trainer_prob / rollout_prob ) per token, in nats — the "
            "exponent of the importance weight."
        ),
        aggregation="per-token.",
        cap=(
            "`BudgetPolicy.clamp = 20.0 nats` (soft) and "
            "`BudgetPolicy.veto_abs_log_ratio = 30.0 nats` (hard)."
        ),
        drop_behavior=(
            "tokens with |log_ratio| > 20 are clipped in place; tokens with "
            "|log_ratio| > 30 fire a veto and quarantine the whole group."
        ),
        long=(
            "Each token's importance weight is exp(log_ratio). "
            "max|log_ratio| is the worst single-token disagreement. A "
            "log_ratio of 1 nat ≈ the trainer is e≈2.7× more confident than "
            "the rollout was."
        ),
    ),
    "sequence_log_ratio": GlossaryEntry(
        what=(
            "Sum of per-token log_ratios across the response, in nats — the "
            "log of the full-sequence importance weight."
        ),
        aggregation="per-sequence (one number per response).",
        cap=(
            "no per-sequence cap; budget action is decided from ESS, "
            "`max_clipped_fraction`, and `veto_abs_log_ratio` on the "
            "constituent tokens."
        ),
        drop_behavior=(
            "no direct drop — propagates into ESS, which drives the budget "
            "decision."
        ),
        long=(
            "How far the rollout drifted from the trainer-side view across "
            "the whole sequence. ±0.5 nats over 128 tokens ≈ negligible; "
            "5+ nats means the engines systematically disagree."
        ),
    ),
    "clipped_fraction": GlossaryEntry(
        what=(
            "Fraction of tokens whose |log_ratio| exceeded the clamp "
            "threshold."
        ),
        aggregation="per-group (fraction over all valid policy tokens).",
        cap=(
            "`BudgetPolicy.max_clipped_fraction = 0.10` (a.k.a. STEER's "
            "`high_clipped_fraction = 0.1`); threshold inside "
            "`BudgetPolicy.clamp = 20.0 nats`."
        ),
        drop_behavior=(
            "above 0.10 routes the group to `train_with_correction`; "
            "individual offending tokens stay clipped in place."
        ),
        long=(
            "Clamping importance weights stops one bad token from blowing "
            "up the gradient. >0.1 here means ≥10% of tokens are clamped — "
            "typically a trigger to mark the group `train_with_correction`."
        ),
    ),
    "veto_fraction": GlossaryEntry(
        what=(
            "Fraction of tokens whose |log_ratio| exceeded the hard-veto "
            "threshold."
        ),
        aggregation="per-group (fraction over all valid policy tokens).",
        cap=(
            "`BudgetPolicy.veto_abs_log_ratio = 30.0 nats`; the veto "
            "fraction itself fires on any non-zero value (veto_fraction > "
            "0.0)."
        ),
        drop_behavior=(
            "any non-zero veto fraction quarantines the entire group; "
            "OPBC does not attempt correction."
        ),
        long=(
            ">0 here means at least one token is so off-policy the OPBC "
            "quarantines the whole group — even after clamping it would "
            "corrupt the gradient."
        ),
    ),
    "top_1pct_gradient_mass": GlossaryEntry(
        what=(
            "Fraction of total importance weight carried by the worst 1% "
            "of tokens."
        ),
        aggregation="per-group (one number per response/group).",
        cap=(
            "no formal cap; rendered alongside `BudgetPolicy.clamp = 20.0 "
            "nats` and `max_clipped_fraction = 0.10` for context."
        ),
        drop_behavior=(
            "no direct drop — used as a concentration diagnostic feeding "
            "operator triage."
        ),
        long=(
            "Tells you whether the drift is uniform (~0.01) or concentrated "
            "in a few outlier tokens (>0.05). Concentrated drift is more "
            "dangerous."
        ),
    ),
    "second_moment": GlossaryEntry(
        what="E[w²] of importance weights w = exp(log_ratio).",
        aggregation="per-group.",
        cap=(
            "no formal cap; reads as the variance upper bound and is "
            "consumed alongside ESS (which is derived from it)."
        ),
        drop_behavior=(
            "no direct drop — feeds the ESS computation, which drives the "
            "group routing decision."
        ),
        long=(
            "Used to bound the variance of the importance-corrected "
            "estimator. Closer to 1 = lower-variance correction."
        ),
    ),
    "router_flip_rate": GlossaryEntry(
        what=(
            "Fraction of (token, layer) pairs where MoE top-1 routed "
            "expert differs between rollout and trainer."
        ),
        aggregation="per-(token, layer); dashboards display the mean.",
        cap=(
            "no cap — observability only. OPBC does not directly act on "
            "router_flip_rate; the metric informs whether to pin the "
            "rollout to the trainer's precision class via `PolicyManifest`."
        ),
        drop_behavior=(
            "no drop — observability only. Decisions to reject a worker on "
            "precision mismatch happen at validator time, not at metric "
            "time."
        ),
        long=(
            "Top-1 stability under quantization or precision changes. Low "
            "values (<5%) suggest the dominant expert is robust. Same as "
            "the top-1 flip rate shown on the router dashboard."
        ),
    ),
    "token_expert_disagreement_rate": GlossaryEntry(
        what=(
            "Fraction of tokens with at least one MoE layer where the "
            "top-k *set* of routed experts differs."
        ),
        aggregation="per-token (a token counts if any layer disagrees).",
        cap="no cap — observability only.",
        drop_behavior="no drop — observability only.",
        long=(
            "More sensitive than top-1 flip — quantization noise often "
            "shuffles the lower-ranked experts even when the dominant one "
            "is stable. Visible in the gap between this and the top-1 flip "
            "rate."
        ),
    ),
    "tool_call_jaccard": GlossaryEntry(
        what=(
            "Jaccard similarity of (tool_name, arguments_hash) sets "
            "between rollout and reference trajectories."
        ),
        aggregation="per-trajectory (one Jaccard value per comparison).",
        cap="no cap — observability only.",
        drop_behavior="no drop — observability only.",
        long=(
            "1.0 = identical tool invocations across the run. <1.0 = the "
            "rollout engine picked at least one tool the reference engine "
            "didn't (or vice versa). Lower = more behavioural drift."
        ),
    ),
    "first_divergence_step": GlossaryEntry(
        what=(
            "First assistant step where rollout and reference trajectories "
            "took a different action."
        ),
        aggregation="per-trajectory (one integer-or-null per comparison).",
        cap="no cap — observability only.",
        drop_behavior="no drop — observability only.",
        long=(
            "If non-null, the rollout's behaviour diverged from the "
            "reference starting at this step. The lower the number, the "
            "earlier the trajectory rolled off-policy."
        ),
    ),
    "answer_match": GlossaryEntry(
        what=(
            "Whether the rollout engine and reference engine reached the "
            "same final assistant answer (normalised whitespace + case)."
        ),
        aggregation=(
            "per-trajectory (boolean); rolled up per engine pair as a rate."
        ),
        cap="no cap — observability only.",
        drop_behavior="no drop — observability only.",
        long=(
            "The headline trajectory metric. Even small per-token drift "
            "can compound into a different final answer when the agent is "
            "making multi-step tool-use decisions."
        ),
    ),
    "bf16": GlossaryEntry(
        what=(
            "16-bit brain-float — the standard training/inference precision "
            "for modern LLMs. Same exponent range as fp32, narrower "
            "mantissa."
        ),
        aggregation=(
            "per-checkpoint (precision_class is a property of how the "
            "engine serves the model, not a per-token measurement)."
        ),
        cap=(
            "no cap — precision class label. Mismatch between worker "
            "precision and `PolicyManifest.precision_class` is rejected at "
            "validator time, not at metric time."
        ),
        drop_behavior=(
            "no drop on its own; downstream `validators.precision_mismatch` "
            "rejects whole groups when the served precision diverges from "
            "the manifest pin."
        ),
        long="",
    ),
    "fp8": GlossaryEntry(
        what=(
            "8-bit floating point. Two formats — E4M3 and E5M2. Halves "
            "memory vs bf16 and runs ~2× faster on Hopper-class GPUs."
        ),
        aggregation="per-checkpoint (precision_class label).",
        cap=(
            "no cap — precision class label. The numerical drift FP8 "
            "introduces is measured by the dense and router dashboards "
            "(see `|Δlogp|`, `router_flip_rate`)."
        ),
        drop_behavior=(
            "no drop on its own; downstream `validators.precision_mismatch` "
            "rejects whole groups when an FP8 worker submits against a "
            "bf16-pinned manifest."
        ),
        long="",
    ),
    # Auxiliary legacy entries (endpoint probe + OPBC narrative).
    # These are not in the STEER rewrite list, but downstream callers
    # (endpoint_dashboard, marketplace_simulation) still reference them.
    # They use the same four-field shape with sentinel values so the
    # test suite stays uniform.
    "top-1 flip rate": GlossaryEntry(
        what=(
            "Alias of `router_flip_rate` — see that entry for cap/drop."
        ),
        aggregation="per-(token, layer).",
        cap="no cap — observability only.",
        drop_behavior="no drop — observability only.",
        long="",
    ),
    "top-k set disagreement": GlossaryEntry(
        what=(
            "Alias of `token_expert_disagreement_rate` — see that entry "
            "for cap/drop."
        ),
        aggregation="per-token.",
        cap="no cap — observability only.",
        drop_behavior="no drop — observability only.",
        long="",
    ),
    "answer_match_rate": GlossaryEntry(
        what=(
            "Rate of `answer_match` across all comparisons in an engine "
            "pair — fraction of tasks where rollout and reference produced "
            "the same final answer."
        ),
        aggregation="per-(rollout_engine, trainer_engine) pair.",
        cap="no cap — observability only.",
        drop_behavior="no drop — observability only.",
        long="",
    ),
    "tool_choice_disagreement_rate": GlossaryEntry(
        what=(
            "Fraction of assistant steps where rollout and reference "
            "invoked different tools."
        ),
        aggregation="per-step, rolled up per trajectory.",
        cap="no cap — observability only.",
        drop_behavior="no drop — observability only.",
        long=(
            "Per-step counterpart to Jaccard. >0 means the trajectories "
            "were divergent at that step level."
        ),
    ),
    "token_ids_available": GlossaryEntry(
        what=(
            "Whether the API exposes the integer token IDs of the response "
            "(not just text)."
        ),
        aggregation="per-endpoint (capability flag).",
        cap="no cap — endpoint capability probe.",
        drop_behavior=(
            "no drop at metric time; missing token IDs make a rollout "
            "unverifiable downstream."
        ),
        long="",
    ),
    "sampled_logprobs_available": GlossaryEntry(
        what="Whether the API returns the logprob of each sampled token.",
        aggregation="per-endpoint (capability flag).",
        cap="no cap — endpoint capability probe.",
        drop_behavior=(
            "no drop at metric time; missing logprobs force the trainer to "
            "redo the forward pass."
        ),
        long="",
    ),
    "top_logprobs_available": GlossaryEntry(
        what=(
            "Whether the API returns the top-k alternatives at each "
            "sampled position."
        ),
        aggregation="per-endpoint (capability flag).",
        cap="no cap — endpoint capability probe.",
        drop_behavior="no drop — capability probe only.",
        long="",
    ),
    "seed_supported": GlossaryEntry(
        what=(
            "Whether the API honours a generation seed and signals it "
            "back via `system_fingerprint`."
        ),
        aggregation="per-endpoint (capability flag).",
        cap="no cap — endpoint capability probe.",
        drop_behavior=(
            "no drop at metric time; missing seed support breaks audit "
            "replay."
        ),
        long="",
    ),
    "OPBC": GlossaryEntry(
        what=(
            "Off-Policy Budget Controller — the project's policy that "
            "decides whether a group is `train`, `train_with_correction`, "
            "`replay`, `quarantine`, or `reject`."
        ),
        aggregation=(
            "per-group (reads per-token / per-group metrics and emits one "
            "typed decision)."
        ),
        cap=(
            "configured by `BudgetPolicy` — `clamp = 20.0 nats`, "
            "`veto_abs_log_ratio = 30.0 nats`, `max_clipped_fraction = "
            "0.10`, `min_ess = 0.30`, `replay_ess_threshold = 0.60`."
        ),
        drop_behavior=(
            "veto → `quarantine`; low ESS → `quarantine`; high "
            "`clipped_fraction` → `train_with_correction`; moderate ESS "
            "or policy lag → `replay`; otherwise → `train`."
        ),
        long="",
    ),
    "TP=4": GlossaryEntry(
        what=(
            "Tensor-parallel size 4 — the model is sharded across 4 GPUs "
            "at the tensor level."
        ),
        aggregation="per-engine deployment.",
        cap="no cap — deployment parameter.",
        drop_behavior="no drop — deployment parameter.",
        long="",
    ),
}


def render_glossary_card(terms: list[str]) -> str:
    """Return a `<section class='card'>` with definitions for the given terms."""
    rows: list[str] = []
    for term in terms:
        entry = GLOSSARY.get(term)
        if entry is None:
            continue
        long_html = (
            f"<p class='gloss-long'>{_html.escape(entry.long)}</p>"
            if entry.long
            else ""
        )
        fields_html = (
            f"<dl class='gloss-fields'>"
            f"<dt>per</dt><dd>{_html.escape(entry.aggregation)}</dd>"
            f"<dt>cap</dt><dd>{_html.escape(entry.cap)}</dd>"
            f"<dt>on cap</dt><dd>{_html.escape(entry.drop_behavior)}</dd>"
            f"</dl>"
        )
        rows.append(
            f"<div class='gloss-item'>"
            f"<div class='gloss-term'><code>{_html.escape(term)}</code></div>"
            f"<div class='gloss-body'>"
            f"<p class='gloss-short'>{_html.escape(entry.what)}</p>"
            f"{fields_html}"
            f"{long_html}"
            f"</div></div>"
        )
    return (
        "<section class='card glossary'>"
        "<h2>What these metrics mean</h2>"
        "<style>"
        ".gloss-item{display:grid;grid-template-columns:230px 1fr;gap:1rem;"
        "align-items:start;padding:.7rem 0;border-bottom:1px solid var(--border)}"
        ".gloss-item:last-child{border-bottom:none}"
        ".gloss-term{min-width:0;overflow-wrap:anywhere;word-break:break-word}"
        ".gloss-term code{font-size:.88rem;background:#eef2ff;color:#3730a3;"
        "padding:.15rem .45rem;border-radius:4px;display:inline-block;"
        "max-width:100%;overflow-wrap:anywhere;line-height:1.35}"
        ".gloss-body{min-width:0}"
        ".gloss-short{margin:0;color:var(--text);font-size:.94rem}"
        ".gloss-fields{display:grid;grid-template-columns:auto 1fr;"
        "column-gap:.6rem;row-gap:.2rem;margin:.45rem 0 .15rem 0;"
        "font-size:.82rem;color:var(--muted);line-height:1.45}"
        ".gloss-fields dt{font-weight:600;color:#475569;text-transform:lowercase}"
        ".gloss-fields dd{margin:0}"
        ".gloss-long{margin:.35rem 0 0 0;color:var(--muted);font-size:.85rem;line-height:1.45}"
        "@media (max-width:580px){.gloss-item{grid-template-columns:1fr;gap:.2rem}}"
        "</style>"
        f"{''.join(rows)}"
        "</section>"
    )


def render_full_glossary_html() -> str:
    """Render a standalone glossary page (used by the public site)."""
    rows: list[str] = []
    for term, entry in GLOSSARY.items():
        long_html = (
            f"<p class='gloss-long'>{_html.escape(entry.long)}</p>"
            if entry.long
            else ""
        )
        fields_html = (
            f"<dl class='gloss-fields'>"
            f"<dt>per</dt><dd>{_html.escape(entry.aggregation)}</dd>"
            f"<dt>cap</dt><dd>{_html.escape(entry.cap)}</dd>"
            f"<dt>on cap</dt><dd>{_html.escape(entry.drop_behavior)}</dd>"
            f"</dl>"
        )
        rows.append(
            f"<div class='gloss-item'>"
            f"<div class='gloss-term'><code>{_html.escape(term)}</code></div>"
            f"<div class='gloss-body'>"
            f"<p class='gloss-short'>{_html.escape(entry.what)}</p>"
            f"{fields_html}"
            f"{long_html}"
            f"</div></div>"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Glossary — streams-rollout-market dashboards</title>"
        "<style>"
        ":root{--bg:#f6f8fb;--surface:#fff;--border:#e2e8f0;--text:#0f172a;"
        "--muted:#64748b;--accent:#2563eb}"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;"
        "background:var(--bg);color:var(--text);max-width:880px;margin:0 auto;"
        "padding:1.5rem 1rem 4rem;line-height:1.5}"
        "header{margin-bottom:1.5rem;text-align:center}"
        "header h1{font-size:1.6rem;margin:0 0 .4rem 0;font-weight:700}"
        "header p{color:var(--muted);margin:0 auto;max-width:600px}"
        "section{background:var(--surface);border:1px solid var(--border);"
        "border-radius:14px;padding:1.25rem 1.75rem;margin:1rem 0}"
        ".gloss-item{display:grid;grid-template-columns:260px 1fr;gap:1rem;"
        "align-items:start;padding:.85rem 0;border-bottom:1px solid var(--border)}"
        ".gloss-item:last-child{border-bottom:none}"
        ".gloss-term{min-width:0;overflow-wrap:anywhere;word-break:break-word}"
        ".gloss-term code{font-size:.88rem;background:#eef2ff;color:#3730a3;"
        "padding:.2rem .55rem;border-radius:5px;display:inline-block;font-weight:500;"
        "max-width:100%;overflow-wrap:anywhere;line-height:1.35}"
        ".gloss-body{min-width:0}"
        ".gloss-short{margin:0;color:var(--text)}"
        ".gloss-fields{display:grid;grid-template-columns:auto 1fr;"
        "column-gap:.7rem;row-gap:.25rem;margin:.5rem 0 .25rem 0;"
        "font-size:.84rem;color:var(--muted);line-height:1.5}"
        ".gloss-fields dt{font-weight:600;color:#475569;text-transform:lowercase}"
        ".gloss-fields dd{margin:0}"
        ".gloss-long{margin:.4rem 0 0 0;color:var(--muted);font-size:.88rem;line-height:1.5}"
        ".back{display:inline-block;margin-bottom:1rem;color:var(--accent);"
        "text-decoration:none;font-size:.9rem}"
        ".back:hover{text-decoration:underline}"
        "@media (max-width:580px){.gloss-item{grid-template-columns:1fr;gap:.2rem}}"
        "</style></head><body>"
        "<a class='back' href='index.html'>← back to dashboards</a>"
        "<header>"
        "<h1>What the metrics mean</h1>"
        "<p>Plain-English definitions of every abbreviation that shows up "
        "in the dashboards, in the four-field shape verl uses for its "
        "train-inference correction module: what the metric is, the unit "
        "it aggregates over, the cap that triggers OPBC action, and what "
        "happens when the cap fires.</p>"
        "</header>"
        "<section>"
        f"{''.join(rows)}"
        "</section>"
        "</body></html>"
    )
