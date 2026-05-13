"""Tests for STEER ``dashboard.graphs_first``.

Asserts the four headline dashboards (dense, router, agent,
marketplace) render with charts/traffic-light tiles up front and the
raw numeric tables tucked under a collapsible ``<details
data-section="raw-numbers">`` block.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from rollout_market.observatory.agent_dashboard import (
    build_dashboard as build_agent_dashboard,
    render_html as render_agent_html,
)
from rollout_market.observatory.agent_trajectory_lab import (
    TrajectoryDivergenceReport,
)
from rollout_market.observatory.dense_dashboard import (
    build_dashboard as build_dense_dashboard,
    render_html as render_dense_html,
)
from rollout_market.observatory.dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
)
from rollout_market.observatory.marketplace_simulation import (
    SimulationResult,
    render_html as render_market_html,
)
from rollout_market.observatory.router_dashboard import (
    build_dashboard as build_router_dashboard,
    render_html as render_router_html,
)
from rollout_market.observatory.router_mismatch_lab import (
    RouterMismatchReport,
)


def _fp(name: str) -> EngineFingerprint:
    return EngineFingerprint(name=name, version="0.0", fingerprint=f"sha256:{name}")


def _dense(idx: int) -> DenseMismatchReport:
    return DenseMismatchReport(
        run_id=f"d{idx}",
        model_id="Qwen/Qwen3-32B",
        checkpoint_digest="sha256:demo",
        tokenizer_hash="tok-x",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        precision_class="bf16",
        prompt_id="p1",
        num_policy_tokens=12,
        delta_logprob_mean=0.0,
        delta_logprob_abs_mean=0.04,
        sequence_log_ratio=0.05,
        ess=0.995,
        second_moment=0.01,
        clipped_fraction=0.0,
        veto_fraction=0.0,
        max_abs_log_ratio=0.5,
        top_1pct_gradient_mass=0.2,
        created_at=datetime(2026, 5, 11, idx, 0, 0, tzinfo=timezone.utc),
    )


def _router(idx: int) -> RouterMismatchReport:
    return RouterMismatchReport(
        run_id=f"r{idx}",
        model_id="Qwen/Qwen3-30B-A3B",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        num_tokens=8,
        num_layers=4,
        top_k=2,
        num_experts=8,
        router_flip_rate=0.05,
        token_expert_disagreement_rate=0.5,
        layer_flip_rates=[0.04, 0.05, 0.06, 0.05],
        created_at=datetime(2026, 5, 11, idx, 0, 0, tzinfo=timezone.utc),
    )


def _agent(idx: int) -> TrajectoryDivergenceReport:
    return TrajectoryDivergenceReport(
        run_id=f"a{idx}",
        task_id=f"t{idx}",
        task_text="demo",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        rollout_step_count=4,
        trainer_step_count=4,
        rollout_assistant_steps=4,
        trainer_assistant_steps=4,
        first_divergence_step=2,
        first_divergence_kind="content",
        tool_choice_disagreement_rate=0.25,
        tool_call_jaccard=0.6,
        final_answer_match=True,
        rollout_terminal_reason="answered",
        trainer_terminal_reason="answered",
        same_step_count=True,
        rollout_tool_signature=[],
        trainer_tool_signature=[],
        created_at=datetime(2026, 5, 11, idx, 0, 0, tzinfo=timezone.utc),
    )


def _market_result() -> SimulationResult:
    return SimulationResult(
        jobs_total=10,
        submissions=10,
        livestore_by_state={"accepted": 6, "rejected": 4},
        rejection_reasons={"precision_mismatch": 2, "tokenizer_mismatch": 2},
        by_profile_action={("honest", "train"): 5, ("toxic", "reject"): 4},
        by_worker_action={("w1", "train"): 5},
        decision_reasons={"within_budget": 5, "veto": 0},
        audit_event_counts={"submitted": 10, "accepted": 6},
    )


def _split_at_raw(page: str) -> tuple[str, str]:
    """Return (headline, raw_block_onwards) split at the <details data-section="raw-numbers">.

    The headline ends at the opening ``<details`` tag of the raw-numbers
    block; that tag and everything after it lives in the raw partition.
    """
    marker = "data-section=\"raw-numbers\""
    idx = page.find(marker)
    assert idx >= 0, "raw-numbers marker missing from rendered page"
    # Walk backward to the opening `<` so the raw partition starts at
    # `<details ...>`.
    open_idx = page.rfind("<", 0, idx)
    assert open_idx >= 0
    return page[:open_idx], page[open_idx:]


def _headline_assertions(page: str) -> None:
    headline, raw = _split_at_raw(page)
    # Headline contains chart markup (Chart.js canvas).
    assert "<canvas" in headline, "headline missing <canvas> chart markup"
    # Marketplace uses chart_block + raw_numbers_block but no traffic-light
    # row; the matrix dashboards add a traffic-light row. Either is fine
    # for the "graphs" assertion.
    assert "<canvas" in headline or "data-chart=" in headline
    # Raw numbers block is a <details> (so it is natively collapsible).
    assert raw.startswith("<details"), (
        f"raw-numbers block must be a <details> element; got {raw[:60]!r}"
    )
    # Raw block carries at least one <table>.
    assert "<table" in raw


def test_dense_dashboard_is_graphs_first():
    dashboard = build_dense_dashboard([_dense(1), _dense(2)])
    page = render_dense_html(dashboard)
    _headline_assertions(page)


def test_router_dashboard_is_graphs_first():
    dashboard = build_router_dashboard([_router(1), _router(2)])
    page = render_router_html(dashboard)
    _headline_assertions(page)


def test_agent_dashboard_is_plain_viewer_not_graphs_first():
    """The agent page is now a plain trajectory viewer — no Chart.js
    canvas, no traffic-light row, no raw-numbers details/table. Those
    were comparison-mode artefacts; the viewer is intentionally
    plain."""
    dashboard = build_agent_dashboard([_agent(1), _agent(2)])
    page = render_agent_html(dashboard, trajectory_dir="/tmp/_nonexistent_agent_dir_")
    assert "Agent trajectory viewer" in page
    assert "<canvas" not in page
    assert "data-chart=\"traffic-light-row\"" not in page


def test_marketplace_simulation_is_graphs_first():
    page = render_market_html(_market_result())
    _headline_assertions(page)


def test_traffic_light_tile_appears_in_headline_dashboards():
    """The matrix dashboards must emit a row of traffic-light tiles.

    The agent page is excluded — it's a plain trajectory viewer now,
    not a comparison matrix.
    """
    for builder, reports, renderer in [
        (build_dense_dashboard, [_dense(1)], render_dense_html),
        (build_router_dashboard, [_router(1)], render_router_html),
    ]:
        dashboard = builder(reports)
        page = renderer(dashboard)
        # `traffic-light-row` is the data-chart marker on the row container.
        assert "data-chart=\"traffic-light-row\"" in page, (
            f"{renderer.__name__} missing traffic-light tiles"
        )
        # Each row carries at least one tile.
        n_tiles = len(re.findall(r"data-chart=\"traffic-light\"", page))
        assert n_tiles >= 1, f"{renderer.__name__} traffic-light row empty"
