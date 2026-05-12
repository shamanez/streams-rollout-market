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


def test_hermes_matrices_appear_before_legacy_matrices() -> None:
    """The headline must lead with Hermes; legacy matrices follow."""
    page = _render()
    hermes_dense = page.find('data-section="hermes-agent-dense-matrix"')
    hermes_moe = page.find('data-section="hermes-agent-moe-matrix"')
    legacy_dense = page.find('data-section="dense-matrix"')
    legacy_moe = page.find('data-section="moe-matrix"')
    assert hermes_dense != -1
    assert hermes_moe != -1
    assert legacy_dense != -1
    assert legacy_moe != -1
    # Hermes matrices precede the legacy ones in source order.
    assert hermes_dense < legacy_dense
    assert hermes_dense < legacy_moe
    assert hermes_moe < legacy_dense
    assert hermes_moe < legacy_moe


def test_legacy_matrices_inside_collapsible_appendix() -> None:
    """The legacy single-prompt matrices must be wrapped in a
    ``<details>`` element so they don't dominate the headline visually."""
    page = _render()
    details_open = page.find("<details class='legacy-archive'")
    details_close = page.find("</details>", details_open)
    assert details_open != -1, "legacy-archive <details> wrapper missing"
    assert details_close != -1
    appendix_html = page[details_open:details_close]
    # Both legacy matrices live inside the appendix.
    assert 'data-section="dense-matrix"' in appendix_html
    assert 'data-section="moe-matrix"' in appendix_html
    # The summary label is operator-facing and signals 'archived'.
    assert "Legacy single-prompt experiments" in appendix_html
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
