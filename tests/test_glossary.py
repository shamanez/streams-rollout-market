"""Tests for STEER ``dashboard.glossary_verl_rewrite``.

Asserts every entry in ``rollout_market.observatory._glossary.GLOSSARY``
carries the four required fields (what, aggregation, cap,
drop_behavior) and that the 15 metrics named in the STEER directive
are present and reference the right ``BudgetPolicy`` / mismatch-metrics
constants where they apply.
"""

from __future__ import annotations

import re

from rollout_market.observatory._glossary import (
    GLOSSARY,
    GlossaryEntry,
    render_full_glossary_html,
    render_glossary_card,
)

# The 15 metrics STEER requires the four-field rewrite to cover.
REQUIRED_TERMS = [
    "ESS",
    "|Δlogp|",
    "log_ratio",
    "sequence_log_ratio",
    "clipped_fraction",
    "veto_fraction",
    "top_1pct_gradient_mass",
    "second_moment",
    "router_flip_rate",
    "token_expert_disagreement_rate",
    "tool_call_jaccard",
    "first_divergence_step",
    "answer_match",
    "bf16",
    "fp8",
]


def test_glossary_is_dict_of_entries():
    assert isinstance(GLOSSARY, dict)
    assert GLOSSARY  # not empty
    for term, entry in GLOSSARY.items():
        assert isinstance(term, str) and term, f"empty term: {term!r}"
        assert isinstance(entry, GlossaryEntry), f"{term!r} is not GlossaryEntry"


def test_every_entry_has_the_four_required_fields():
    """STEER mandates: every entry has (1) what, (2) aggregation,
    (3) cap, (4) drop_behavior — all non-empty strings."""
    for term, entry in GLOSSARY.items():
        assert isinstance(entry.what, str) and entry.what.strip(), (
            f"{term!r} missing field 1: what"
        )
        assert isinstance(entry.aggregation, str) and entry.aggregation.strip(), (
            f"{term!r} missing field 2: aggregation"
        )
        assert isinstance(entry.cap, str) and entry.cap.strip(), (
            f"{term!r} missing field 3: cap"
        )
        assert isinstance(entry.drop_behavior, str) and entry.drop_behavior.strip(), (
            f"{term!r} missing field 4: drop_behavior"
        )


def test_required_steer_terms_are_present():
    missing = [t for t in REQUIRED_TERMS if t not in GLOSSARY]
    assert not missing, f"STEER-required terms missing from GLOSSARY: {missing}"


def test_cap_field_references_named_constants_where_applicable():
    """Metrics that OPBC acts on must name the constant carrying the cap."""
    # Per STEER directive: clamp=20.0 nats, veto_abs_log_ratio=30.0 nats,
    # high_clipped_fraction=0.1 (a.k.a. max_clipped_fraction), veto_fraction=0.0.
    must_mention_clamp = ["|Δlogp|", "log_ratio", "clipped_fraction"]
    for term in must_mention_clamp:
        cap = GLOSSARY[term].cap
        assert "clamp" in cap.lower() and "20" in cap, (
            f"{term!r} cap should reference BudgetPolicy.clamp = 20.0 nats; got {cap!r}"
        )
    must_mention_veto = ["log_ratio", "veto_fraction"]
    for term in must_mention_veto:
        cap = GLOSSARY[term].cap
        assert "veto" in cap.lower() and "30" in cap, (
            f"{term!r} cap should reference BudgetPolicy.veto_abs_log_ratio = 30.0; got {cap!r}"
        )
    # clipped_fraction must mention the 0.10 threshold by name.
    cf = GLOSSARY["clipped_fraction"].cap
    assert ("0.10" in cf or "0.1" in cf), (
        f"clipped_fraction cap should mention 0.10 / max_clipped_fraction; got {cf!r}"
    )
    # ESS caps reference the OPBC routing thresholds.
    ess_cap = GLOSSARY["ESS"].cap
    assert "ess" in ess_cap.lower() or "0.30" in ess_cap or "0.60" in ess_cap, (
        f"ESS cap should reference replay/min ESS thresholds; got {ess_cap!r}"
    )


def test_drop_behavior_uses_verl_vocabulary():
    """STEER directive lists two drop behaviors verbatim:
    'token clipped in place' vs 'entire group quarantined'.
    Each metric that has a cap must use one of those phrases."""
    drop_keywords = ("clip", "quarantin", "no drop", "no direct drop", "no cap")
    for term, entry in GLOSSARY.items():
        text = entry.drop_behavior.lower()
        assert any(k in text for k in drop_keywords), (
            f"{term!r} drop_behavior must use verl vocabulary "
            f"(clipped / quarantined / no drop): {entry.drop_behavior!r}"
        )


def test_render_glossary_card_emits_all_four_fields():
    html = render_glossary_card(REQUIRED_TERMS)
    # All four field labels appear in the rendered card.
    assert "per" in html
    assert "cap" in html
    assert "on cap" in html
    # Each required term is present in the card body.
    for term in REQUIRED_TERMS:
        # |Δlogp| gets HTML-escaped (& is special); just check the bare term.
        # Other terms are alphanumeric and render as-is.
        if term == "|Δlogp|":
            assert "|" in html and "logp" in html
        else:
            assert term in html, f"{term!r} missing from glossary card"


def test_render_full_glossary_html_contains_every_entry():
    html = render_full_glossary_html()
    assert "<!doctype html>" in html
    # Every entry appears at least once (term and what).
    for term, entry in GLOSSARY.items():
        if term == "|Δlogp|":
            assert "logp" in html
        else:
            # Strip equals sign for "TP=4" — html escapes are fine.
            cleaned = re.sub(r"[^a-zA-Z0-9_ ]", "", term)
            assert cleaned in html or term in html, f"{term!r} missing"
        # The first sentence of `what` shows up.
        snippet = entry.what.split(".")[0][:40]
        assert snippet in html, f"first sentence of {term!r} missing"
