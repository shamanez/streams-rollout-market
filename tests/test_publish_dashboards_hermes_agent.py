"""Tests for the Hermes-Agent matrix sections in publish_dashboards.py.

Covers STEER.md's render acceptance gate:
- The rendered docs/index.html MUST contain
  data-section="hermes-agent-dense-matrix" with 2 mx-tile children
  whose mx-value text is numeric and NOT in {—, TBD, 0%, 0.00, n/a, ""}
- Each tile's colour class MUST be one of {mx-good, mx-warn, mx-bad}
  (NOT mx-empty/mx-tbd)
- The 8 pre-existing matrix tiles (Dense × 4 + MoE × 4) must continue
  to render with at least one canonical value (e.g. 0.9987 or 4.3%)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_dashboards", REPO_ROOT / "scripts" / "live" / "publish_dashboards.py"
)
publish_dashboards = importlib.util.module_from_spec(SPEC)
sys.modules["publish_dashboards"] = publish_dashboards
SPEC.loader.exec_module(publish_dashboards)  # type: ignore[union-attr]


# --- fixture payloads -------------------------------------------------------


def _dense_payload_with_canonical_ess() -> dict:
    """Mock dense payload that hits the canonical 0.9987 / 0.9994 / 0.9949 /
    0.9948 cells from PROGRESS.md so the non-regression test passes."""
    return {
        "rows": [
            {
                "model_id": "Qwen/Qwen3-32B",
                "rollout_engine": "vllm-bf16",
                "trainer_engine": "fsdp",
                "precision_class": "bf16",
                "device_bucket": "L40S (g6e.12xlarge)",
                "ess": 0.9987,
            },
            {
                "model_id": "Qwen/Qwen3-32B",
                "rollout_engine": "vllm-fp8",
                "trainer_engine": "fsdp",
                "precision_class": "fp8",
                "device_bucket": "L40S (g6e.12xlarge)",
                "ess": 0.9949,
            },
            {
                "model_id": "Qwen/Qwen3-32B",
                "rollout_engine": "vllm-bf16",
                "trainer_engine": "megatron-lm",
                "precision_class": "bf16",
                "device_bucket": "L40S (g6e.12xlarge)",
                "ess": 0.9994,
            },
            {
                "model_id": "Qwen/Qwen3-32B",
                "rollout_engine": "vllm-fp8",
                "trainer_engine": "megatron-lm",
                "precision_class": "fp8",
                "device_bucket": "L40S (g6e.12xlarge)",
                "ess": 0.9948,
            },
        ]
    }


def _router_payload_with_canonical_flips() -> dict:
    return {
        "rows": [
            {
                "model_id": "Qwen/Qwen3-30B-A3B",
                "rollout_engine": "vllm-bf16",
                "trainer_engine": "fsdp",
                "precision_class": "bf16",
                "device_bucket": "L40S (g6e.12xlarge)",
                "router_flip_rate": 0.043,
            },
            {
                "model_id": "Qwen/Qwen3-30B-A3B",
                "rollout_engine": "vllm-fp8",
                "trainer_engine": "fsdp",
                "precision_class": "fp8",
                "device_bucket": "L40S (g6e.12xlarge)",
                "router_flip_rate": 0.071,
            },
            {
                "model_id": "Qwen/Qwen3-30B-A3B",
                "rollout_engine": "vllm-bf16",
                "trainer_engine": "megatron-lm",
                "precision_class": "bf16",
                "device_bucket": "L40S (g6e.12xlarge)",
                "router_flip_rate": 0.041,
            },
            {
                "model_id": "Qwen/Qwen3-30B-A3B",
                "rollout_engine": "vllm-fp8",
                "trainer_engine": "megatron-lm",
                "precision_class": "fp8",
                "device_bucket": "L40S (g6e.12xlarge)",
                "router_flip_rate": 0.063,
            },
        ]
    }


def _agent_payload_full_matrix() -> dict:
    """An agent_dashboard payload with both dense + MoE engine pairs filled
    in for the four traffic-light Hermes tiles."""
    return {
        "engine_pairs": [
            {
                "rollout_engine": "hermes-qwen3-32b-fp8",
                "trainer_engine": "hermes-qwen3-32b-bf16",
                "count": 12,
                "mean_first_divergence_step": 1.4,
                "fully_matched_count": 9,
                "mean_tool_call_jaccard": 0.61,
                "final_answer_match_rate": 0.75,
            },
            {
                "rollout_engine": "hermes-qwen3-32b-bf16_seedB",
                "trainer_engine": "hermes-qwen3-32b-bf16",
                "count": 12,
                "mean_first_divergence_step": 2.6,
                "fully_matched_count": 11,
                "mean_tool_call_jaccard": 0.88,
                "final_answer_match_rate": 0.92,
            },
            {
                "rollout_engine": "hermes-qwen3-30b-a3b-fp8",
                "trainer_engine": "hermes-qwen3-30b-a3b-bf16",
                "count": 12,
                "mean_first_divergence_step": 1.1,
                "fully_matched_count": 7,
                "mean_tool_call_jaccard": 0.52,
                "final_answer_match_rate": 0.58,
            },
            {
                "rollout_engine": "hermes-qwen3-30b-a3b-bf16_seedB",
                "trainer_engine": "hermes-qwen3-30b-a3b-bf16",
                "count": 12,
                "mean_first_divergence_step": 2.4,
                "fully_matched_count": 10,
                "mean_tool_call_jaccard": 0.84,
                "final_answer_match_rate": 0.83,
            },
        ]
    }


def _card_data_with_hermes() -> list[dict]:
    cards = [{**c, "ready": True} for c in publish_dashboards.CARDS]
    for c in cards:
        if c["slug"] == "dense":
            c["payload"] = _dense_payload_with_canonical_ess()
        elif c["slug"] == "router":
            c["payload"] = _router_payload_with_canonical_flips()
        elif c["slug"] == "agent":
            c["payload"] = _agent_payload_full_matrix()
        else:
            c["payload"] = {}
        c["filename"] = f"{c['slug']}_dashboard.html"
    return cards


# --- structural assertions --------------------------------------------------


def test_hermes_dense_matrix_section_present() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    assert 'data-section="hermes-agent-dense-matrix"' in page


def test_hermes_moe_matrix_section_present() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    assert 'data-section="hermes-agent-moe-matrix"' in page


def test_hermes_dense_matrix_has_two_tiles() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    section_start = page.index('data-section="hermes-agent-dense-matrix"')
    section_end = page.index("</section>", section_start)
    section = page[section_start:section_end]
    # Each tile is an <a class="mx-tile mx-...">; count those inside this section.
    tile_count = section.count('class="mx-tile mx-')
    assert tile_count == 2, f"expected 2 mx-tile children in hermes-dense section, got {tile_count}"


def test_hermes_moe_matrix_has_two_tiles() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    section_start = page.index('data-section="hermes-agent-moe-matrix"')
    section_end = page.index("</section>", section_start)
    section = page[section_start:section_end]
    tile_count = section.count('class="mx-tile mx-')
    assert tile_count == 2, f"expected 2 mx-tile children in hermes-moe section, got {tile_count}"


# --- numeric mx-value + valid colour class ----------------------------------


def _extract_tile_blocks(section: str) -> list[str]:
    """Split an mx-section into per-tile substrings."""
    out = []
    i = 0
    while True:
        j = section.find('class="mx-tile mx-', i)
        if j == -1:
            break
        k = section.find("</a>", j)
        if k == -1:
            break
        out.append(section[j : k + len("</a>")])
        i = k
    return out


_BAD_PLACEHOLDERS = {"—", "TBD", "0%", "0.00", "n/a", "", "—/—"}


def _mx_value(tile: str) -> str:
    needle = "<div class='mx-value'>"
    start = tile.find(needle)
    if start == -1:
        return ""
    start += len(needle)
    end = tile.find("</div>", start)
    return tile[start:end] if end != -1 else ""


def _mx_class(tile: str) -> str:
    needle = 'class="mx-tile mx-'
    start = tile.find(needle) + len(needle)
    end = tile.find('"', start)
    return tile[start:end] if end != -1 else ""


_COUNT_RE_PATTERN = r"^\d+/\d+$"


def _looks_like_answered_count(v: str) -> bool:
    """The mx-value is the answered/total count `X/N`."""
    import re
    return bool(re.match(_COUNT_RE_PATTERN, v)) and "/0" not in v.split("/", 1)[-1].lstrip()


def test_hermes_dense_tiles_have_numeric_values_and_valid_class() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    a = page.index('data-section="hermes-agent-dense-matrix"')
    b = page.index("</section>", a)
    section = page[a:b]
    tiles = _extract_tile_blocks(section)
    assert tiles, "no tiles found in hermes-dense section"
    for tile in tiles:
        v = _mx_value(tile)
        assert v not in _BAD_PLACEHOLDERS, f"placeholder value rendered: {v!r}"
        # mx-value is an "X/N" answered-of-total count.
        assert _looks_like_answered_count(v), (
            f"hermes tile value must be answered-count X/N, got {v!r}"
        )
        cls = _mx_class(tile)
        assert cls in {"good", "warn", "bad"}, (
            f"tile colour class must be good/warn/bad, got mx-{cls!r}"
        )


def test_hermes_moe_tiles_have_numeric_values_and_valid_class() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    a = page.index('data-section="hermes-agent-moe-matrix"')
    b = page.index("</section>", a)
    section = page[a:b]
    tiles = _extract_tile_blocks(section)
    assert tiles
    for tile in tiles:
        v = _mx_value(tile)
        assert v not in _BAD_PLACEHOLDERS
        assert _looks_like_answered_count(v), (
            f"hermes tile value must be answered-count X/N, got {v!r}"
        )
        cls = _mx_class(tile)
        assert cls in {"good", "warn", "bad"}


# --- non-regression: pre-existing tiles still render ------------------------


def test_dense_matrix_still_renders_canonical_ess() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    assert 'data-section="dense-matrix"' in page
    # At least one canonical ESS value from PROGRESS.md must appear.
    assert any(v in page for v in ("0.9987", "0.9994", "0.9949", "0.9948"))


def test_moe_matrix_still_renders_canonical_flip_rates() -> None:
    page = publish_dashboards.render_index(_card_data_with_hermes())
    assert 'data-section="moe-matrix"' in page
    # At least one canonical router_flip_rate from PROGRESS.md must appear.
    assert any(v in page for v in ("4.3%", "4.1%", "7.1%", "6.3%"))


# --- empty / partial payload handling ---------------------------------------


def test_hermes_section_renders_empty_tiles_when_no_data() -> None:
    """When the agent dashboard payload is empty, the section still renders
    its 2 tiles (as mx-empty-tile placeholders) — the structural shape stays
    stable so the test can assert on it before the live data lands."""
    cards = _card_data_with_hermes()
    for c in cards:
        if c["slug"] == "agent":
            c["payload"] = {"engine_pairs": []}
    page = publish_dashboards.render_index(cards)
    a = page.index('data-section="hermes-agent-dense-matrix"')
    b = page.index("</section>", a)
    section = page[a:b]
    # mx-empty-tile is a valid (non-data) tile class.
    tile_count = section.count("mx-tile mx-empty-tile") + section.count("mx-tile mx-good") + section.count("mx-tile mx-warn") + section.count("mx-tile mx-bad")
    assert tile_count == 2


def test_select_hermes_pairs_picks_precision_and_noise_floor() -> None:
    """The _select_hermes_pairs helper must correctly identify the
    fp8-vs-bf16 (precision-effect) and bf16_seedB-vs-bf16 (noise-floor)
    pairs from an agent_dashboard payload."""
    payload = _agent_payload_full_matrix()
    p, n = publish_dashboards._select_hermes_pairs(
        payload,
        model_substring="qwen3-32b",
        bf16_engine="hermes-qwen3-32b-bf16",
        fp8_engine="hermes-qwen3-32b-fp8",
        seedb_engine="hermes-qwen3-32b-bf16_seedB",
    )
    assert p is not None and p["rollout_engine"] == "hermes-qwen3-32b-fp8"
    assert n is not None and n["rollout_engine"] == "hermes-qwen3-32b-bf16_seedB"
