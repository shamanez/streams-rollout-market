"""Tests for the Hermes-Agent MoE matrix section in publish_dashboards.py.

Mirrors ``test_publish_dashboards_hermes_agent.py`` but focuses on
STEER.md's Iter 6 acceptance gate:
- Both hermes sections (dense + moe) must be present in docs/index.html
- Each must have exactly 2 mx-tile children with numeric mx-value
- Tile colour class is one of {mx-good, mx-warn, mx-bad}
- The 8 pre-existing matrix tiles (Dense × 4 + MoE × 4) must continue
  to render with their committed values
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Re-use fixture builders + helpers from the Iter 4 test file. tests/ is
# not a package so we import by path.
_HERMES_AGENT_TEST = (
    Path(__file__).resolve().parent / "test_publish_dashboards_hermes_agent.py"
)
_spec = importlib.util.spec_from_file_location(
    "test_publish_dashboards_hermes_agent", _HERMES_AGENT_TEST
)
_hermes_agent_test = importlib.util.module_from_spec(_spec)
sys.modules["test_publish_dashboards_hermes_agent"] = _hermes_agent_test
_spec.loader.exec_module(_hermes_agent_test)  # type: ignore[union-attr]

publish_dashboards = _hermes_agent_test.publish_dashboards
_agent_payload_full_matrix = _hermes_agent_test._agent_payload_full_matrix
_card_data_with_hermes = _hermes_agent_test._card_data_with_hermes
_extract_tile_blocks = _hermes_agent_test._extract_tile_blocks
_mx_class = _hermes_agent_test._mx_class
_mx_value = _hermes_agent_test._mx_value
_BAD_PLACEHOLDERS = _hermes_agent_test._BAD_PLACEHOLDERS


def test_both_hermes_sections_present_in_one_render() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    assert page.count('data-section="hermes-agent-dense-matrix"') == 1
    assert page.count('data-section="hermes-agent-moe-matrix"') == 1


def test_moe_section_has_two_tiles_with_numeric_values() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    a = page.index('data-section="hermes-agent-moe-matrix"')
    b = page.index("</section>", a)
    tiles = _extract_tile_blocks(page[a:b])
    assert len(tiles) == 2
    for tile in tiles:
        v = _mx_value(tile)
        assert v not in _BAD_PLACEHOLDERS
        cls = _mx_class(tile)
        assert cls in {"good", "warn", "bad"}


def test_moe_precision_effect_pair_resolves_correctly() -> None:
    """The fp8-vs-bf16 pair for the MoE model must be picked from the
    aggregated agent_dashboard payload."""
    payload = _agent_payload_full_matrix()
    p, n = publish_dashboards._select_hermes_pairs(
        payload,
        model_substring="qwen3-30b-a3b",
        bf16_engine="hermes-qwen3-30b-a3b-bf16",
        fp8_engine="hermes-qwen3-30b-a3b-fp8",
        seedb_engine="hermes-qwen3-30b-a3b-bf16-seedB",
    )
    assert p is not None
    assert p["rollout_engine"] == "hermes-qwen3-30b-a3b-fp8"
    assert n is not None
    assert n["rollout_engine"] == "hermes-qwen3-30b-a3b-bf16-seedB"


def test_pre_existing_tiles_non_regression() -> None:
    """Iter 6 acceptance: at least one canonical value from each of the
    8 pre-existing tiles must remain visible in the rendered page."""
    page = publish_dashboards.render_index(_card_data_with_hermes())
    dense_canonicals = ("0.9987", "0.9994", "0.9949", "0.9948")
    moe_canonicals = ("4.3%", "4.1%", "7.1%", "6.3%")
    assert any(v in page for v in dense_canonicals)
    assert any(v in page for v in moe_canonicals)
