"""Endpoint identity-gap dashboard.

Aggregates many endpoint_contract_report.json files (each produced by
``rollout_market.observatory.endpoint_probe``) into one HTML+JSON
dashboard. The dashboard answers a single question:

    For each provider/model pair we have probed, which token/logprob
    contract fields does the endpoint actually expose?

Capability flags are summarised three ways:
  - per provider × model row
  - per capability column (across all rows)
  - a single coverage score per row (fraction of capabilities present)

The dashboard is read-only; it never re-probes any endpoint. Run the
endpoint_probe CLI first to populate the runs directory.
"""

from __future__ import annotations

import glob as _glob
import html as _html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .endpoint_probe import EndpointContractReport

CAPABILITY_FIELDS: tuple[str, ...] = (
    "token_ids_available",
    "sampled_logprobs_available",
    "top_logprobs_available",
    "seed_supported",
)


@dataclass(frozen=True)
class EndpointRow:
    """One (provider, model_label) entry on the dashboard."""

    provider: str
    model_label: str
    token_ids_available: bool
    sampled_logprobs_available: bool
    top_logprobs_available: bool
    seed_supported: bool
    response_status: int
    error: str | None
    raw_response_hash: str
    last_probed_at: str

    @property
    def coverage_score(self) -> float:
        present = sum(int(getattr(self, f)) for f in CAPABILITY_FIELDS)
        return present / len(CAPABILITY_FIELDS)


@dataclass
class EndpointDashboard:
    rows: list[EndpointRow] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.rows)

    def capability_totals(self) -> dict[str, int]:
        totals = {f: 0 for f in CAPABILITY_FIELDS}
        for row in self.rows:
            for f in CAPABILITY_FIELDS:
                if getattr(row, f):
                    totals[f] += 1
        return totals

    def providers_seen(self) -> list[str]:
        return sorted({r.provider for r in self.rows})

    def models_seen(self) -> list[str]:
        return sorted({r.model_label for r in self.rows})

    def as_dict(self) -> dict[str, object]:
        return {
            "num_runs": self.num_runs,
            "providers": self.providers_seen(),
            "models": self.models_seen(),
            "capability_totals": self.capability_totals(),
            "rows": [
                {
                    "provider": r.provider,
                    "model_label": r.model_label,
                    "token_ids_available": r.token_ids_available,
                    "sampled_logprobs_available": r.sampled_logprobs_available,
                    "top_logprobs_available": r.top_logprobs_available,
                    "seed_supported": r.seed_supported,
                    "response_status": r.response_status,
                    "error": r.error,
                    "raw_response_hash": r.raw_response_hash,
                    "last_probed_at": r.last_probed_at,
                    "coverage_score": r.coverage_score,
                }
                for r in self.rows
            ],
        }


def _row_from_report(report: EndpointContractReport) -> EndpointRow:
    return EndpointRow(
        provider=report.provider,
        model_label=report.model_label,
        token_ids_available=report.token_ids_available,
        sampled_logprobs_available=report.sampled_logprobs_available,
        top_logprobs_available=report.top_logprobs_available,
        seed_supported=report.seed_supported,
        response_status=report.response_status,
        error=report.error,
        raw_response_hash=report.raw_response_hash,
        last_probed_at=report.ran_at.isoformat(),
    )


def _latest_per_pair(rows: list[EndpointRow]) -> list[EndpointRow]:
    """Keep only the latest row for each (provider, model_label) key."""
    by_key: dict[tuple[str, str], EndpointRow] = {}
    for row in rows:
        key = (row.provider, row.model_label)
        existing = by_key.get(key)
        if existing is None or row.last_probed_at > existing.last_probed_at:
            by_key[key] = row
    return sorted(by_key.values(), key=lambda r: (r.provider, r.model_label))


def build_dashboard(reports: Iterable[EndpointContractReport]) -> EndpointDashboard:
    """Aggregate an iterable of probe reports into one dashboard.

    When the same (provider, model_label) pair appears more than once, the
    *latest* report (by ran_at) wins. Older runs are intentionally
    discarded so the dashboard reflects current state, not history.
    """
    rows = [_row_from_report(r) for r in reports]
    return EndpointDashboard(rows=_latest_per_pair(rows))


def load_reports_from_paths(paths: Iterable[Path | str]) -> list[EndpointContractReport]:
    """Load probe reports from explicit file paths."""
    reports: list[EndpointContractReport] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        reports.append(EndpointContractReport.model_validate_json(text))
    return reports


def load_reports_glob(pattern: str) -> list[EndpointContractReport]:
    """Load every endpoint_contract_report.json that matches the glob."""
    return load_reports_from_paths(sorted(_glob.glob(pattern, recursive=True)))


def render_html(dashboard: EndpointDashboard) -> str:
    """Render a self-contained dashboard page (no JS, no remote assets)."""

    def _yes_no(flag: bool) -> str:
        return "✓" if flag else "—"

    capability_header = "".join(
        f"<th>{_html.escape(f)}</th>" for f in CAPABILITY_FIELDS
    )
    rows_html = ""
    for row in dashboard.rows:
        cells = "".join(
            f"<td>{_yes_no(getattr(row, f))}</td>" for f in CAPABILITY_FIELDS
        )
        rows_html += (
            f"<tr><td>{_html.escape(row.provider)}</td>"
            f"<td>{_html.escape(row.model_label)}</td>"
            f"{cells}"
            f"<td>{row.response_status}</td>"
            f"<td>{row.coverage_score:.2f}</td>"
            f"<td>{_html.escape(row.error or '')}</td></tr>"
        )
    totals = dashboard.capability_totals()
    total_rows = "".join(
        f"<tr><td>{_html.escape(k)}</td><td>{v}/{dashboard.num_runs}</td></tr>"
        for k, v in totals.items()
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Endpoint identity-gap dashboard</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:920px;margin:2rem auto;color:#111}"
        "table{border-collapse:collapse;width:100%;margin-bottom:1rem}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5}"
        "h1{font-size:1.2rem}h2{font-size:1rem;margin-top:1.5rem}"
        "</style></head><body>"
        f"<h1>Endpoint identity-gap dashboard ({dashboard.num_runs} run"
        f"{'' if dashboard.num_runs == 1 else 's'})</h1>"
        "<h2>Coverage by provider × model</h2>"
        "<table><thead><tr>"
        "<th>provider</th><th>model_label</th>"
        f"{capability_header}"
        "<th>status</th><th>coverage</th><th>error</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
        "<h2>Capability totals</h2>"
        "<table><thead><tr><th>capability</th><th>present</th></tr></thead>"
        f"<tbody>{total_rows}</tbody></table>"
        "</body></html>"
    )


def write_dashboard(dashboard: EndpointDashboard, out_dir: Path | str) -> dict[str, Path]:
    """Persist JSON + HTML side by side under out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "endpoint_dashboard.json"
    html_path = out_path / "endpoint_dashboard.html"
    json_path.write_text(json.dumps(dashboard.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html(dashboard), encoding="utf-8")
    return {"json": json_path, "html": html_path}
