"""Tests for STEER ``dashboard.engine_filter``.

Asserts that the dense, router, and agent dashboards split their
rendered HTML into a *headline* zone (vLLM × FSDP/Megatron only) and
a collapsible *All engines (full data)* appendix that contains
everything else (sglang as rollout, HF as either side). Ingestion is
unaffected — only display changes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rollout_market.observatory._engine_filter import (
    APPENDIX_HEADER,
    is_headline_pair,
)
from rollout_market.observatory.agent_dashboard import (
    AgentDashboard,
    build_dashboard as build_agent_dashboard,
    render_html as render_agent_html,
)
from rollout_market.observatory.agent_trajectory_lab import (
    TrajectoryDivergenceReport,
)
from rollout_market.observatory.dense_dashboard import (
    DenseDashboard,
    build_dashboard as build_dense_dashboard,
    render_html as render_dense_html,
)
from rollout_market.observatory.dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
)
from rollout_market.observatory.router_dashboard import (
    RouterDashboard,
    build_dashboard as build_router_dashboard,
    render_html as render_router_html,
)
from rollout_market.observatory.router_mismatch_lab import (
    RouterMismatchReport,
)


def _fp(name: str) -> EngineFingerprint:
    return EngineFingerprint(name=name, version="0.0", fingerprint=f"sha256:{name}")


def test_is_headline_pair_classification():
    assert is_headline_pair("vllm-bf16", "fsdp")
    assert is_headline_pair("vllm", "megatron-lm")
    assert is_headline_pair("vLLM-FP8", "FSDP")
    assert not is_headline_pair("sglang", "fsdp")
    assert not is_headline_pair("vllm", "hf-transformers")
    assert not is_headline_pair("sglang", "hf-transformers")
    assert not is_headline_pair("", "")


def _dense_report(rollout: str, trainer: str, *, ess: float, idx: int) -> DenseMismatchReport:
    return DenseMismatchReport(
        run_id=f"r{idx}-{rollout}-{trainer}",
        model_id="Qwen/Qwen3-32B",
        checkpoint_digest="sha256:demo",
        tokenizer_hash="tok-x",
        rollout_engine=_fp(rollout),
        trainer_engine=_fp(trainer),
        precision_class="bf16",
        prompt_id="p1",
        num_policy_tokens=12,
        delta_logprob_mean=0.0,
        delta_logprob_abs_mean=0.04,
        sequence_log_ratio=0.05,
        ess=ess,
        second_moment=0.01,
        clipped_fraction=0.0,
        veto_fraction=0.0,
        max_abs_log_ratio=0.5,
        top_1pct_gradient_mass=0.2,
        created_at=datetime(2026, 5, 11, idx, 0, 0, tzinfo=timezone.utc),
    )


def _split(page: str, marker: str) -> tuple[str, str]:
    """Split a rendered page at the appendix marker.

    The appendix is wrapped in a ``<details data-section='all-engines'>``
    block; everything before it is the headline zone.
    """
    idx = page.find(marker)
    assert idx >= 0, f"appendix marker {marker!r} missing from page"
    return page[:idx], page[idx:]


def test_dense_dashboard_purges_blocked_engines_from_html():
    """Inference-only redirect: sglang and HF rows are absent from the
    rendered HTML (no appendix, no headline), but still appear in
    `as_dict()` for JSON consumers."""
    reports = [
        _dense_report("vllm", "fsdp", ess=0.999, idx=1),
        _dense_report("vllm", "megatron", ess=0.997, idx=2),
        _dense_report("sglang", "fsdp", ess=0.984, idx=3),
        _dense_report("vllm", "hf-transformers", ess=0.991, idx=4),
        _dense_report("sglang", "hf-transformers", ess=0.973, idx=5),
    ]
    dashboard = build_dense_dashboard(reports)
    page = render_dense_html(dashboard)
    # vLLM × {FSDP, Megatron} reach the rendered HTML.
    assert "fsdp" in page.lower()
    assert "megatron" in page.lower()
    assert "vllm" in page.lower()
    # sglang and HF are purged from rendering entirely.
    assert "sglang" not in page.lower()
    assert "hf-transformers" not in page.lower()
    # No appendix block.
    assert "data-section=\"all-engines\"" not in page
    assert APPENDIX_HEADER not in page
    # Ingestion is unchanged: all 5 rows survive in JSON.
    payload = dashboard.as_dict()
    assert payload["num_runs"] == 5
    assert len(payload["engine_pairs"]) == 5


def _router_report(rollout: str, trainer: str, *, idx: int) -> RouterMismatchReport:
    return RouterMismatchReport(
        run_id=f"r{idx}-{rollout}-{trainer}",
        model_id="Qwen/Qwen3-30B-A3B",
        rollout_engine=_fp(rollout),
        trainer_engine=_fp(trainer),
        num_tokens=8,
        num_layers=4,
        top_k=2,
        num_experts=8,
        router_flip_rate=0.05,
        token_expert_disagreement_rate=0.5,
        layer_flip_rates=[0.04, 0.05, 0.06, 0.05],
        created_at=datetime(2026, 5, 11, idx, 0, 0, tzinfo=timezone.utc),
    )


def test_router_dashboard_purges_blocked_engines_from_html():
    reports = [
        _router_report("vllm", "fsdp", idx=1),
        _router_report("vllm", "megatron", idx=2),
        _router_report("sglang", "fsdp", idx=3),
        _router_report("vllm", "hf-transformers", idx=4),
    ]
    dashboard = build_router_dashboard(reports)
    page = render_router_html(dashboard)
    assert "fsdp" in page.lower()
    assert "megatron" in page.lower()
    assert "vllm" in page.lower()
    assert "sglang" not in page.lower()
    assert "hf-transformers" not in page.lower()
    assert "data-section=\"all-engines\"" not in page
    payload = dashboard.as_dict()
    assert payload["num_runs"] == 4
    assert len(payload["engine_pairs"]) == 4


def _agent_report(rollout: str, trainer: str, *, idx: int) -> TrajectoryDivergenceReport:
    return TrajectoryDivergenceReport(
        run_id=f"r{idx}-{rollout}-{trainer}",
        task_id=f"task-{idx}",
        task_text="demo",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_fp(rollout),
        trainer_engine=_fp(trainer),
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


def test_agent_dashboard_json_still_holds_engine_pairs():
    """The HTML side became a plain viewer (no engine comparison), but
    the underlying JSON dashboard still aggregates engine pairs for
    consumers of the JSON contract. This test guards the JSON shape
    only — the HTML purge invariant no longer applies."""
    reports = [
        _agent_report("vllm", "fsdp", idx=1),
        _agent_report("vllm", "megatron", idx=2),
        _agent_report("sglang", "fsdp", idx=3),
        _agent_report("vllm", "hf-transformers", idx=4),
        _agent_report("sglang", "hf-transformers", idx=5),
    ]
    dashboard = build_agent_dashboard(reports)
    payload = dashboard.as_dict()
    assert payload["num_runs"] == 5
    assert len(payload["engine_pairs"]) == 5


def test_empty_dashboards_render_without_appendix():
    """An empty dashboard still emits the headline skeleton and no
    appendix. Only the comparison matrices (dense/router) are checked
    here — the agent page is a plain viewer with a different
    structure."""
    for renderer, empty in [
        (render_dense_html, DenseDashboard()),
        (render_router_html, RouterDashboard()),
    ]:
        page = renderer(empty)
        assert "data-section=\"headline-engines\"" in page
        assert "data-section=\"all-engines\"" not in page
    # Agent page: empty viewer shows the empty-state card.
    agent_page = render_agent_html(
        AgentDashboard(), trajectory_dir="/tmp/_nonexistent_agent_dir_"
    )
    assert "data-section=\"trajectories\"" in agent_page
    assert "Captured trajectories (0)" in agent_page
