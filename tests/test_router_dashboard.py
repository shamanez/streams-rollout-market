from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rollout_market.cli import router_dashboard as cli
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint
from rollout_market.observatory.router_dashboard import (
    RouterDashboard,
    build_dashboard,
    load_reports_glob,
    render_html,
    write_dashboard,
)
from rollout_market.observatory.router_mismatch_lab import (
    RouterMismatchReport,
    write_router_mismatch,
)


def _fp(name: str) -> EngineFingerprint:
    return EngineFingerprint(name=name, version="0.0", fingerprint=f"sha256:{name}")


def _report(
    *,
    run_id: str = "r0",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    rollout: str = "vllm",
    trainer: str = "fsdp",
    num_tokens: int = 4,
    num_layers: int = 3,
    top_k: int = 2,
    num_experts: int = 8,
    router_flip_rate: float = 0.10,
    token_expert_disagreement_rate: float = 0.25,
    layer_flip_rates: list[float] | None = None,
    created_at: datetime | None = None,
) -> RouterMismatchReport:
    if layer_flip_rates is None:
        layer_flip_rates = [0.05, 0.10, 0.15]
    return RouterMismatchReport(
        run_id=run_id,
        model_id=model_id,
        rollout_engine=_fp(rollout),
        trainer_engine=_fp(trainer),
        num_tokens=num_tokens,
        num_layers=num_layers,
        top_k=top_k,
        num_experts=num_experts,
        router_flip_rate=router_flip_rate,
        token_expert_disagreement_rate=token_expert_disagreement_rate,
        layer_flip_rates=layer_flip_rates,
        created_at=created_at or datetime(2026, 5, 10, tzinfo=timezone.utc),
    )


def test_build_dashboard_keeps_every_run():
    dashboard = build_dashboard([_report(run_id="r1"), _report(run_id="r2")])
    assert dashboard.num_runs == 2


def test_rows_sorted_by_created_at():
    early = _report(run_id="early", created_at=datetime(2026, 5, 9, tzinfo=timezone.utc))
    late = _report(run_id="late", created_at=datetime(2026, 5, 11, tzinfo=timezone.utc))
    middle = _report(
        run_id="middle", created_at=datetime(2026, 5, 10, tzinfo=timezone.utc)
    )
    dashboard = build_dashboard([late, early, middle])
    assert [r.run_id for r in dashboard.rows] == ["early", "middle", "late"]


def test_layer_flip_rate_summary():
    dashboard = build_dashboard(
        [_report(layer_flip_rates=[0.0, 0.5, 0.25])]
    )
    row = dashboard.rows[0]
    assert row.layer_flip_rate_min == 0.0
    assert row.layer_flip_rate_max == 0.5
    assert row.layer_flip_rate_mean == 0.25


def test_engine_pair_aggregate_means():
    reports = [
        _report(run_id="a", router_flip_rate=0.10, token_expert_disagreement_rate=0.20),
        _report(run_id="b", router_flip_rate=0.30, token_expert_disagreement_rate=0.40),
    ]
    dashboard = build_dashboard(reports)
    aggs = dashboard.engine_pair_aggregates()
    assert len(aggs) == 1
    agg = aggs[0]
    assert agg.count == 2
    assert agg.mean_router_flip_rate == pytest.approx(0.20)
    assert agg.mean_token_expert_disagreement_rate == pytest.approx(0.30)
    assert agg.layer_count == 3


def test_engine_pair_separates_distinct_pairs():
    dashboard = build_dashboard(
        [
            _report(run_id="a", rollout="vllm", trainer="fsdp"),
            _report(run_id="b", rollout="vllm", trainer="megatron"),
        ]
    )
    keys = {(a.rollout_engine, a.trainer_engine) for a in dashboard.engine_pair_aggregates()}
    assert keys == {("vllm", "fsdp"), ("vllm", "megatron")}


def test_engine_pair_worst_layer_flip_rate():
    dashboard = build_dashboard(
        [
            _report(layer_flip_rates=[0.0, 0.1, 0.2]),
            _report(layer_flip_rates=[0.0, 0.6, 0.0]),
        ]
    )
    agg = dashboard.engine_pair_aggregates()[0]
    assert agg.worst_layer_flip_rate == 0.6


def test_engine_pair_total_tokens():
    dashboard = build_dashboard(
        [_report(num_tokens=10), _report(num_tokens=15)]
    )
    agg = dashboard.engine_pair_aggregates()[0]
    assert agg.total_tokens == 25


def test_render_html_includes_pair_and_run_tables():
    dashboard = build_dashboard([_report(run_id="r1")])
    page = render_html(dashboard)
    assert page.startswith("<!doctype html>")
    assert "Per (rollout_engine -> trainer_engine) pair" in page
    assert "Per run" in page
    assert "r1" in page
    assert "vllm" in page


def test_render_html_escapes_user_input():
    dashboard = build_dashboard(
        [_report(run_id="<svg/onload=alert(1)>", model_id="<x>", rollout="vllm-<y>")]
    )
    page = render_html(dashboard)
    assert "<svg/onload" not in page
    assert "&lt;svg" in page
    assert "&lt;y&gt;" in page


def test_write_dashboard_creates_both_files(tmp_path: Path):
    dashboard = build_dashboard([_report()])
    paths = write_dashboard(dashboard, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["html"].exists()


def test_load_reports_glob_reads_files(tmp_path: Path):
    runs_root = tmp_path / "runs"
    base_time = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(3):
        write_router_mismatch(
            _report(run_id=f"r{i}", created_at=base_time + timedelta(seconds=i)),
            runs_root / f"sub{i}",
        )
    reports = load_reports_glob(str(runs_root / "*" / "router_mismatch_report.json"))
    assert len(reports) == 3


def test_empty_dashboard_renders():
    dashboard = RouterDashboard()
    page = render_html(dashboard)
    assert "<!doctype html>" in page
    # Headline skeleton renders even with no rows; the appendix block
    # has been purged from public dashboards.
    assert "data-section=\"headline-engines\"" in page
    assert "data-section=\"all-engines\"" not in page


def test_cli_main_writes_dashboard(tmp_path: Path):
    runs_root = tmp_path / "runs"
    base_time = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(2):
        write_router_mismatch(
            _report(run_id=f"r{i}", created_at=base_time + timedelta(seconds=i)),
            runs_root / f"sub{i}",
        )
    rc = cli.main(
        [
            "--reports-glob",
            str(runs_root / "*" / "router_mismatch_report.json"),
            "--out-dir",
            str(tmp_path / "dashboard"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "dashboard" / "router_dashboard.json").exists()
    assert (tmp_path / "dashboard" / "router_dashboard.html").exists()


def test_dashboard_as_dict_round_trip():
    dashboard = build_dashboard([_report(run_id="r1"), _report(run_id="r2")])
    payload = dashboard.as_dict()
    assert payload["num_runs"] == 2
    assert len(payload["rows"]) == 2
    assert "engine_pairs" in payload
