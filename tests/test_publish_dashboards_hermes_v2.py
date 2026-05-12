"""Hermes-first headline + legacy-appendix structure tests.

Cycle 3 v2 (STEER.md `hermes_agent.matrix_render_and_codex` setup
phase) restructures the published dashboard so the Hermes-Agent
multi-turn matrices carry the headline and the legacy single-prompt
matrices move into a collapsible ``<details>`` appendix below them.

This is the *structural* half of the directive — the full
acceptance gate (4 hermes-Dense + 4 hermes-MoE tiles populated with
real numeric values, codex review PASS) requires real Hermes-Agent
trajectory reports from the spot run. These tests assert only what
the render does on its own: ordering, the appendix's collapsible
shape, and that the legacy matrices remain *reachable* below the fold.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_dashboards_v2", REPO_ROOT / "scripts" / "live" / "publish_dashboards.py"
)
publish_dashboards = importlib.util.module_from_spec(SPEC)
sys.modules["publish_dashboards_v2"] = publish_dashboards
SPEC.loader.exec_module(publish_dashboards)  # type: ignore[union-attr]


def _render() -> str:
    card_data = [{**c, "ready": False, "payload": {}} for c in publish_dashboards.CARDS]
    return publish_dashboards.render_index(card_data)


def test_hermes_matrices_appear_before_legacy_dense_archive() -> None:
    """The headline must lead with Hermes; the legacy Dense matrix
    follows in a collapsible appendix.

    Operator direction 2026-05-12 ('Remove all the numbers from cycle-2
    MoE'): the cycle-2 single-prompt MoE router_flip_rate matrix is
    no longer rendered in the legacy appendix at all — it's been
    replaced with the ``router-flip-rate-placeholder`` section above
    the appendix. Only the cycle-2 Dense ESS matrix survives in the
    archive (ESS is still a token-prob-shift metric per the pivot).
    """
    page = _render()
    hermes_dense = page.find('data-section="hermes-agent-dense-matrix"')
    hermes_moe = page.find('data-section="hermes-agent-moe-matrix"')
    legacy_dense = page.find('data-section="dense-matrix"')
    placeholder = page.find('data-section="router-flip-rate-placeholder"')
    assert hermes_dense != -1
    assert hermes_moe != -1
    assert legacy_dense != -1
    assert placeholder != -1
    # Cycle-2 MoE router_flip_rate matrix retired entirely.
    assert 'data-section="moe-matrix"' not in page
    # Hermes matrices precede the placeholder + the legacy Dense archive
    # in source order.
    assert hermes_dense < placeholder
    assert hermes_moe < placeholder
    assert placeholder < legacy_dense


def test_legacy_dense_inside_collapsible_appendix() -> None:
    """The remaining cycle-2 Dense matrix must be wrapped in a
    ``<details>`` element so it doesn't dominate the headline visually."""
    page = _render()
    details_open = page.find("<details class='legacy-archive'")
    details_close = page.find("</details>", details_open)
    assert details_open != -1, "legacy-archive <details> wrapper missing"
    assert details_close != -1
    appendix_html = page[details_open:details_close]
    # Cycle-2 Dense matrix lives inside the appendix.
    assert 'data-section="dense-matrix"' in appendix_html
    # Cycle-2 MoE matrix must NOT live inside the appendix (or anywhere).
    assert 'data-section="moe-matrix"' not in appendix_html
    # The summary label is operator-facing and signals 'archived'.
    assert "Legacy single-prompt Dense matrix" in appendix_html
    # And carries the data-section hook so downstream tests can target it.
    head = page[details_open : details_open + 200]
    assert (
        'data-section="legacy-single-prompt"' in head
        or "data-section='legacy-single-prompt'" in head
    )


def test_hermes_matrices_outside_legacy_appendix() -> None:
    """The Hermes matrices must NOT be wrapped in the legacy appendix —
    they are the headline, not archived content."""
    page = _render()
    details_open = page.find("<details class='legacy-archive'")
    appendix = page[details_open:]
    assert 'data-section="hermes-agent-dense-matrix"' not in appendix
    assert 'data-section="hermes-agent-moe-matrix"' not in appendix


def test_router_flip_rate_placeholder_visible() -> None:
    """The expert-routing 'TBD' slot must render in the headline area
    (NOT hidden behind <details>) so readers know the metric is coming.

    Operator direction 2026-05-12: cycle-2 MoE numbers are wrong; add a
    placeholder for the expert-flopping work. The placeholder must
    document the why (vLLM 0.20.x OpenAI shim limitation) so a future
    operator can pick it up from the dashboard alone.
    """
    page = _render()
    placeholder = page.find('data-section="router-flip-rate-placeholder"')
    details_open = page.find("<details class='legacy-archive'")
    assert placeholder != -1, "router-flip-rate placeholder section missing"
    # Placeholder sits OUTSIDE the collapsible legacy archive (not hidden).
    assert placeholder < details_open
    # Placeholder content names the metric + the why.
    section_end = page.find("</section>", placeholder)
    section_html = page[placeholder:section_end]
    assert "Router flip rate" in section_html
    assert "TBD" in section_html
    # Documents the vLLM root cause so a future operator can pick it up.
    assert "routed_experts" in section_html
    assert "vLLM" in section_html
