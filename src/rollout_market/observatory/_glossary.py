"""Plain-English explanations of every dashboard metric.

The dashboards are dense with abbreviations (ESS, |Δlogp|, top-k flip,
Jaccard...). Each render_html embeds the relevant subset of these
entries inline so the visitor doesn't have to leave the page.
"""

from __future__ import annotations

import html as _html

# Each entry: short term -> (one-line definition, optional longer context)
GLOSSARY: dict[str, tuple[str, str]] = {
    "ESS": (
        "Effective sample size: how usable a rollout is for off-policy training.",
        "ESS = (Σw)² / (N · Σw²) where w is the importance weight per token. "
        "ESS=1 means the rollout matches the trainer's policy exactly. "
        "ESS dropping toward 0 means the rollout is increasingly off-policy "
        "and a trainer would need stronger correction (or skip the rollout).",
    ),
    "|Δlogp|": (
        "Per-token disagreement size between rollout and trainer logprobs (in nats).",
        "Mean over the response of |trainer_logp(token) − rollout_logp(token)|. "
        "Tiny values (~0.01) mean the two engines agree on most tokens; "
        "values > 0.1 mean meaningful single-token drift.",
    ),
    "log_ratio": (
        "log( trainer_prob / rollout_prob ) per token, in nats.",
        "Each token's importance weight is exp(log_ratio). max|log_ratio| "
        "is the worst single-token disagreement. log_ratio of 1 nat ≈ "
        "the trainer is e≈2.7× more confident than the rollout was.",
    ),
    "sequence_log_ratio": (
        "Sum of per-token log_ratios across the response, in nats.",
        "How far the rollout drifted from the trainer-side view across "
        "the whole sequence. ±0.5 nats over 128 tokens ≈ negligible; "
        "5+ nats means the engines systematically disagree.",
    ),
    "clipped_fraction": (
        "Fraction of tokens whose |log_ratio| exceeded the clamp threshold (typically 20).",
        "Clamping importance weights stops one bad token from blowing up the "
        "gradient. >0.1 here means ≥10% of tokens are clamped — typically a "
        "trigger to mark the group `train_with_correction`.",
    ),
    "veto_fraction": (
        "Fraction of tokens whose |log_ratio| exceeded the hard-veto threshold (typically 30).",
        ">0 here means at least one token is so off-policy the OPBC quarantines "
        "the whole group — even after clamping it would corrupt the gradient.",
    ),
    "top_1pct_gradient_mass": (
        "Fraction of total importance weight carried by the worst 1% of tokens.",
        "Tells you whether the drift is uniform (~0.01) or concentrated in a "
        "few outlier tokens (>0.05). Concentrated drift is more dangerous.",
    ),
    "second_moment": (
        "E[w²] of importance weights w = exp(log_ratio).",
        "Used to bound the variance of the importance-corrected estimator. "
        "Closer to 1 = lower-variance correction.",
    ),
    "top-1 flip rate": (
        "Fraction of (token, layer) pairs where MoE top-1 routed expert differs "
        "between rollout and trainer.",
        "Top-1 stability under quantization or precision changes. Low values "
        "(<5%) suggest the dominant expert is robust.",
    ),
    "top-k set disagreement": (
        "Fraction of tokens with at least one MoE layer where the top-k *set* "
        "of experts differs.",
        "More sensitive than top-1 flip — quantization noise often shuffles "
        "the lower-ranked experts even when the dominant one is stable. "
        "Visible in the gap between this and the top-1 flip rate.",
    ),
    "router_flip_rate": (
        "Same as top-1 flip rate.",
        "",
    ),
    "token_expert_disagreement_rate": (
        "Same as top-k set disagreement.",
        "",
    ),
    "first_divergence_step": (
        "First assistant step where two trajectories took a different action.",
        "If non-null, the rollout's behaviour diverged from the reference "
        "starting at this step. The lower the number, the earlier the "
        "trajectory rolled off-policy.",
    ),
    "tool_call_jaccard": (
        "Jaccard similarity of (tool_name, arguments_hash) sets between two trajectories.",
        "1.0 = identical tool invocations across the run. <1.0 = the rollout "
        "engine picked at least one tool the reference engine didn't (or vice "
        "versa). Lower = more behavioural drift.",
    ),
    "tool_choice_disagreement_rate": (
        "Fraction of assistant steps where rollout and reference invoked different tools.",
        "Per-step counterpart to Jaccard. >0 means the trajectories were "
        "divergent at that step level.",
    ),
    "answer_match_rate": (
        "Fraction of tasks where the rollout engine and reference engine reached "
        "the same final assistant answer (normalised whitespace + case).",
        "The headline trajectory metric. Even small per-token drift can compound "
        "into a different final answer when the agent is making multi-step "
        "tool-use decisions.",
    ),
    "token_ids_available": (
        "Whether the API exposes the integer token IDs of the response (not just text).",
        "Required for verifiable rollouts: a trainer can't recompute logprobs "
        "without the token IDs the rollout was scored on.",
    ),
    "sampled_logprobs_available": (
        "Whether the API returns the logprob of each sampled token.",
        "If the rollout side doesn't expose logprobs, the trainer can't compute "
        "the importance ratio and must redo the forward pass entirely.",
    ),
    "top_logprobs_available": (
        "Whether the API returns the top-k alternatives at each sampled position.",
        "Needed for some off-policy correction schemes that average over the "
        "top-k. Nice-to-have, not strictly required.",
    ),
    "seed_supported": (
        "Whether the API honours a generation seed and signals it back via "
        "system_fingerprint.",
        "Without seed support, the rollout is non-reproducible — the trainer "
        "can't audit a problematic group by re-running the same prompt.",
    ),
    "OPBC": (
        "Off-Policy Budget Controller — the project's policy that decides whether "
        "a group is `train`, `train_with_correction`, `replay`, `quarantine`, "
        "or `reject`.",
        "Reads the budget metrics above (ESS, clipped_fraction, etc.) plus "
        "policy-pin checks and emits a typed decision with reasons.",
    ),
    "bf16": (
        "16-bit brain-float — the standard training/inference precision for "
        "modern LLMs. Same exponent range as fp32, narrower mantissa.",
        "",
    ),
    "FP8": (
        "8-bit floating point. Two formats — E4M3 and E5M2. Halves memory "
        "vs bf16 and runs ~2× faster on Hopper-class GPUs.",
        "Cost: introduces measurable numerical drift relative to bf16, which "
        "is exactly what the dense and router dashboards measure.",
    ),
    "TP=4": (
        "Tensor-parallel size 4 — the model is sharded across 4 GPUs at the "
        "tensor level (e.g. each attention head split across 4 devices).",
        "",
    ),
}


def render_glossary_card(terms: list[str]) -> str:
    """Return a `<section class='card'>` with definitions for the given terms."""
    rows = []
    for term in terms:
        if term not in GLOSSARY:
            continue
        short, long_ = GLOSSARY[term]
        long_html = (
            f"<p class='gloss-long'>{_html.escape(long_)}</p>" if long_ else ""
        )
        rows.append(
            f"<div class='gloss-item'>"
            f"<div class='gloss-term'><code>{_html.escape(term)}</code></div>"
            f"<div class='gloss-body'>"
            f"<p class='gloss-short'>{_html.escape(short)}</p>"
            f"{long_html}"
            f"</div></div>"
        )
    return (
        "<section class='card glossary'>"
        "<h2>What these metrics mean</h2>"
        "<style>"
        ".gloss-item{display:grid;grid-template-columns:160px 1fr;gap:1rem;"
        "padding:.7rem 0;border-bottom:1px solid var(--border)}"
        ".gloss-item:last-child{border-bottom:none}"
        ".gloss-term code{font-size:.92rem;background:#eef2ff;color:#3730a3;"
        "padding:.15rem .45rem;border-radius:4px;display:inline-block}"
        ".gloss-short{margin:0;color:var(--text);font-size:.94rem}"
        ".gloss-long{margin:.35rem 0 0 0;color:var(--muted);font-size:.85rem;line-height:1.45}"
        "@media (max-width:580px){.gloss-item{grid-template-columns:1fr;gap:.2rem}}"
        "</style>"
        f"{''.join(rows)}"
        "</section>"
    )


def render_full_glossary_html() -> str:
    """Render a standalone glossary page (used by the public site)."""
    rows = []
    for term, (short, long_) in GLOSSARY.items():
        long_html = f"<p class='gloss-long'>{_html.escape(long_)}</p>" if long_ else ""
        rows.append(
            f"<div class='gloss-item'>"
            f"<div class='gloss-term'><code>{_html.escape(term)}</code></div>"
            f"<div class='gloss-body'>"
            f"<p class='gloss-short'>{_html.escape(short)}</p>"
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
        ".gloss-item{display:grid;grid-template-columns:170px 1fr;gap:1rem;"
        "padding:.85rem 0;border-bottom:1px solid var(--border)}"
        ".gloss-item:last-child{border-bottom:none}"
        ".gloss-term code{font-size:.92rem;background:#eef2ff;color:#3730a3;"
        "padding:.2rem .55rem;border-radius:5px;display:inline-block;font-weight:500}"
        ".gloss-short{margin:0;color:var(--text)}"
        ".gloss-long{margin:.4rem 0 0 0;color:var(--muted);font-size:.88rem;line-height:1.5}"
        ".back{display:inline-block;margin-bottom:1rem;color:var(--accent);"
        "text-decoration:none;font-size:.9rem}"
        ".back:hover{text-decoration:underline}"
        "@media (max-width:580px){.gloss-item{grid-template-columns:1fr;gap:.2rem}}"
        "</style></head><body>"
        "<a class='back' href='index.html'>← back to dashboards</a>"
        "<header>"
        "<h1>What the metrics mean</h1>"
        "<p>Plain-English definitions of every abbreviation that shows up in "
        "the dashboards. The short line at the top of each entry is enough "
        "for a quick scan; the dimmer line underneath gives the longer "
        "context.</p>"
        "</header>"
        "<section>"
        f"{''.join(rows)}"
        "</section>"
        "</body></html>"
    )
