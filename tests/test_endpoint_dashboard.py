from datetime import datetime, timezone
from pathlib import Path

from rollout_market.cli import endpoint_dashboard as cli
from rollout_market.observatory.endpoint_dashboard import (
    EndpointDashboard,
    build_dashboard,
    load_reports_glob,
    render_html,
    write_dashboard,
)
from rollout_market.observatory.endpoint_probe import EndpointContractReport, write_report


def _report(
    *,
    provider: str,
    model_label: str,
    token_ids: bool = False,
    sampled_logprobs: bool = True,
    top_logprobs: bool = False,
    seed_supported: bool = False,
    status: int = 200,
    error: str | None = None,
    ran_at: datetime | None = None,
) -> EndpointContractReport:
    return EndpointContractReport(
        provider=provider,
        model_label=model_label,
        base_url="https://api.example.com/v1",
        sampling_config={"temperature": 0.7},
        sampling_config_hash="sha256:smp",
        request_hash="sha256:req",
        response_status=status,
        response_byte_size=42,
        raw_response_hash="sha256:resp",
        token_ids_available=token_ids,
        sampled_logprobs_available=sampled_logprobs,
        top_logprobs_available=top_logprobs,
        seed_supported=seed_supported,
        tokenizer_id=None,
        system_fingerprint=None,
        error=error,
        ran_at=ran_at or datetime.now(timezone.utc),
    )


def test_dashboard_aggregates_capability_totals():
    dashboard = build_dashboard(
        [
            _report(provider="groq", model_label="llama", sampled_logprobs=True),
            _report(provider="nvidia", model_label="qwen3", sampled_logprobs=False),
            _report(provider="cerebras", model_label="llama", top_logprobs=True),
        ]
    )
    totals = dashboard.capability_totals()
    assert totals["sampled_logprobs_available"] == 2
    assert totals["top_logprobs_available"] == 1
    assert totals["token_ids_available"] == 0


def test_dashboard_keeps_latest_per_pair():
    older = _report(
        provider="groq",
        model_label="llama",
        sampled_logprobs=False,
        ran_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )
    newer = _report(
        provider="groq",
        model_label="llama",
        sampled_logprobs=True,
        top_logprobs=True,
        ran_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
    )
    dashboard = build_dashboard([older, newer])
    assert dashboard.num_runs == 1
    assert dashboard.rows[0].sampled_logprobs_available is True
    assert dashboard.rows[0].top_logprobs_available is True


def test_dashboard_coverage_score():
    dashboard = build_dashboard(
        [
            _report(
                provider="groq",
                model_label="llama",
                token_ids=True,
                sampled_logprobs=True,
                top_logprobs=True,
                seed_supported=True,
            )
        ]
    )
    assert dashboard.rows[0].coverage_score == 1.0


def test_dashboard_lists_unique_providers_and_models():
    dashboard = build_dashboard(
        [
            _report(provider="groq", model_label="llama"),
            _report(provider="groq", model_label="qwen"),
            _report(provider="nvidia", model_label="qwen"),
        ]
    )
    assert dashboard.providers_seen() == ["groq", "nvidia"]
    assert dashboard.models_seen() == ["llama", "qwen"]


def test_render_html_marks_present_capabilities():
    dashboard = build_dashboard(
        [
            _report(
                provider="groq",
                model_label="llama",
                sampled_logprobs=True,
                top_logprobs=False,
            )
        ]
    )
    page = render_html(dashboard)
    assert page.startswith("<!doctype html>")
    assert "groq" in page
    assert "llama" in page
    assert "Coverage matrix" in page or "Capability coverage" in page


def test_render_html_escapes_user_input():
    dashboard = build_dashboard(
        [_report(provider="<svg/onload=alert(1)>", model_label="m", error="<img src=x>")]
    )
    page = render_html(dashboard)
    # Chart.js gets injected via <script> tags; we instead assert that
    # user-controlled HTML payloads are escaped.
    assert "<svg/onload=alert(1)>" not in page
    assert "&lt;svg/onload=alert(1)&gt;" in page
    assert "&lt;img src=x&gt;" in page


def test_write_dashboard_creates_both_files(tmp_path: Path):
    dashboard = build_dashboard([_report(provider="p", model_label="m")])
    paths = write_dashboard(dashboard, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["html"].exists()


def test_load_reports_glob_reads_files(tmp_path: Path):
    runs_root = tmp_path / "runs"
    for i in range(3):
        sub = runs_root / f"r{i}"
        write_report(
            _report(provider=f"p{i}", model_label="m", ran_at=datetime(2026, 5, 10 + i, tzinfo=timezone.utc)),
            sub,
        )
    reports = load_reports_glob(str(runs_root / "*" / "endpoint_contract_report.json"))
    assert len(reports) == 3


def test_empty_dashboard_renders():
    dashboard = EndpointDashboard()
    page = render_html(dashboard)
    assert "0 runs" in page or "(0" in page or "Endpoint identity-gap" in page
    assert "<table>" in page


def test_cli_main_writes_dashboard(tmp_path: Path):
    runs_root = tmp_path / "runs"
    for i in range(2):
        write_report(
            _report(provider=f"p{i}", model_label="m"),
            runs_root / f"r{i}",
        )
    rc = cli.main(
        [
            "--reports-glob",
            str(runs_root / "*" / "endpoint_contract_report.json"),
            "--out-dir",
            str(tmp_path / "dashboard"),
        ]
    )
    assert rc == 0
    json_path = tmp_path / "dashboard" / "endpoint_dashboard.json"
    html_path = tmp_path / "dashboard" / "endpoint_dashboard.html"
    assert json_path.exists()
    assert html_path.exists()
