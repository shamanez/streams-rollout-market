from datetime import datetime, timedelta, timezone
from pathlib import Path

from rollout_market.cli import dense_dashboard as cli
from rollout_market.observatory.dense_dashboard import (
    DenseDashboard,
    build_dashboard,
    load_reports_glob,
    render_html,
    write_dashboard,
)
from rollout_market.observatory.dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
    write_dense_mismatch,
)


def _fp(name: str) -> EngineFingerprint:
    return EngineFingerprint(name=name, version="0.0", fingerprint=f"sha256:{name}")


def _report(
    *,
    run_id: str = "r0",
    model_id: str = "Qwen/Qwen3-32B",
    rollout: str = "vllm",
    trainer: str = "transformers",
    precision: str = "bf16",
    ess: float = 0.95,
    clipped: float = 0.01,
    veto: float = 0.0,
    max_abs: float = 0.5,
    seq_log_ratio: float = 0.05,
    delta_abs: float = 0.04,
    top1pct: float = 0.2,
    created_at: datetime | None = None,
) -> DenseMismatchReport:
    return DenseMismatchReport(
        run_id=run_id,
        model_id=model_id,
        checkpoint_digest="sha256:demo",
        tokenizer_hash="tok-x",
        rollout_engine=_fp(rollout),
        trainer_engine=_fp(trainer),
        precision_class=precision,
        prompt_id="p1",
        num_policy_tokens=12,
        delta_logprob_mean=0.0,
        delta_logprob_abs_mean=delta_abs,
        sequence_log_ratio=seq_log_ratio,
        ess=ess,
        second_moment=1.0,
        clipped_fraction=clipped,
        veto_fraction=veto,
        max_abs_log_ratio=max_abs,
        top_1pct_gradient_mass=top1pct,
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


def test_engine_pair_aggregate_means():
    reports = [
        _report(run_id="a", rollout="vllm", trainer="transformers", ess=0.9, clipped=0.0),
        _report(run_id="b", rollout="vllm", trainer="transformers", ess=0.7, clipped=0.2),
    ]
    dashboard = build_dashboard(reports)
    aggs = dashboard.engine_pair_aggregates()
    assert len(aggs) == 1
    agg = aggs[0]
    assert agg.count == 2
    assert agg.mean_ess == 0.8
    assert agg.mean_clipped_fraction == 0.1


def test_engine_pair_separates_distinct_pairs():
    dashboard = build_dashboard(
        [
            _report(run_id="a", rollout="vllm", trainer="transformers"),
            _report(run_id="b", rollout="sglang", trainer="transformers"),
        ]
    )
    aggs = {(a.rollout_engine, a.trainer_engine): a for a in dashboard.engine_pair_aggregates()}
    assert ("vllm", "transformers") in aggs
    assert ("sglang", "transformers") in aggs


def test_engine_pair_worst_max_abs_log_ratio():
    dashboard = build_dashboard(
        [
            _report(run_id="a", max_abs=1.0),
            _report(run_id="b", max_abs=4.5),
            _report(run_id="c", max_abs=2.0),
        ]
    )
    agg = dashboard.engine_pair_aggregates()[0]
    assert agg.worst_max_abs_log_ratio == 4.5


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
        [_report(run_id="<svg/onload=alert(1)>", model_id="<x>", rollout="<y>")]
    )
    page = render_html(dashboard)
    assert "<svg/onload" not in page
    assert "&lt;svg" in page
    assert "&lt;x&gt;" in page


def test_write_dashboard_creates_both_files(tmp_path: Path):
    dashboard = build_dashboard([_report()])
    paths = write_dashboard(dashboard, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["html"].exists()


def test_load_reports_glob_reads_files(tmp_path: Path):
    runs_root = tmp_path / "runs"
    for i in range(3):
        write_dense_mismatch(
            _report(run_id=f"r{i}", created_at=datetime(2026, 5, 10, i, tzinfo=timezone.utc)),
            runs_root / f"sub{i}",
        )
    reports = load_reports_glob(str(runs_root / "*" / "dense_mismatch_report.json"))
    assert len(reports) == 3


def test_empty_dashboard_renders():
    dashboard = DenseDashboard()
    page = render_html(dashboard)
    assert "<!doctype html>" in page
    assert "0 run" in page or "Per (rollout_engine" in page


def test_cli_main_writes_dashboard(tmp_path: Path):
    runs_root = tmp_path / "runs"
    base_time = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(2):
        write_dense_mismatch(
            _report(run_id=f"r{i}", created_at=base_time + timedelta(seconds=i)),
            runs_root / f"sub{i}",
        )
    rc = cli.main(
        [
            "--reports-glob",
            str(runs_root / "*" / "dense_mismatch_report.json"),
            "--out-dir",
            str(tmp_path / "dashboard"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "dashboard" / "dense_dashboard.json").exists()
    assert (tmp_path / "dashboard" / "dense_dashboard.html").exists()


def test_engine_pair_total_tokens():
    dashboard = build_dashboard(
        [_report(run_id="a"), _report(run_id="b")]
    )
    agg = dashboard.engine_pair_aggregates()[0]
    assert agg.total_policy_tokens == 24
