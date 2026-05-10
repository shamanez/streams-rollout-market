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

from ._dashboard_style import (
    badge,
    chart_block,
    kpi_block,
    page_shell,
    palette_for,
)
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
    totals = dashboard.capability_totals()
    n = dashboard.num_runs

    kpis = kpi_block([
        (str(n), "endpoints probed"),
        (f"{totals.get('sampled_logprobs_available', 0)}/{n}", "expose sampled_logprobs"),
        (f"{totals.get('top_logprobs_available', 0)}/{n}", "expose top_logprobs"),
        (f"{totals.get('seed_supported', 0)}/{n}", "honour seed"),
        (f"{totals.get('token_ids_available', 0)}/{n}", "expose token_ids"),
    ])

    cap_chart = chart_block(
        canvas_id="endpoint-coverage",
        chart_type="bar",
        data={
            "labels": list(CAPABILITY_FIELDS),
            "datasets": [{
                "label": "Endpoints exposing this capability",
                "data": [totals.get(f, 0) for f in CAPABILITY_FIELDS],
                "backgroundColor": palette_for(len(CAPABILITY_FIELDS)),
                "borderRadius": 6,
            }],
        },
        options={
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": f"Out of {n} probed endpoints"}},
            "scales": {"y": {"min": 0, "max": max(n, 1)}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    )

    # Coverage matrix as a colour-coded grid.
    grid_rows: list[str] = []
    for row in dashboard.rows:
        cells = []
        for f in CAPABILITY_FIELDS:
            present = bool(getattr(row, f))
            cells.append(
                f"<td style='text-align:center;background:{'#dcfce7' if present else '#fee2e2'};color:{'#16a34a' if present else '#dc2626'};font-weight:600'>"
                f"{'✓' if present else '✗'}</td>"
            )
        if row.response_status == 200:
            status_cell = badge(str(row.response_status), "good")
        elif row.response_status in (400, 403, 404, 429):
            status_cell = badge(str(row.response_status), "bad")
        else:
            status_cell = badge(str(row.response_status), "warn")
        grid_rows.append(
            "<tr>"
            f"<td><strong>{_html.escape(row.provider)}</strong></td>"
            f"<td><code>{_html.escape(row.model_label)}</code></td>"
            f"{''.join(cells)}"
            f"<td>{status_cell}</td>"
            f"<td>{row.coverage_score:.2f}</td>"
            f"<td>{_html.escape(row.error or '')}</td>"
            "</tr>"
        )

    body = (
        f'<section class="card">{kpis}</section>'
        f'<section class="card">'
        f"<h2>Capability coverage across endpoints</h2>"
        f"<p class='sub'>How many of the {n} probed endpoints expose each contract field a trainer needs. "
        "More green = more endpoints are usable as trainable rollout sources.</p>"
        f"{cap_chart}"
        f"</section>"
        f'<section class="card">'
        f"<h2>Coverage matrix</h2>"
        f"<p class='sub'>One row per <code>(provider, model_label)</code> pair. Latest probe wins on duplicates.</p>"
        "<table><thead><tr>"
        "<th>provider</th><th>model_label</th>"
        + "".join(f"<th>{_html.escape(f)}</th>" for f in CAPABILITY_FIELDS)
        + "<th>status</th><th>coverage</th><th>error</th>"
        "</tr></thead>"
        f"<tbody>{''.join(grid_rows)}</tbody></table>"
        f"</section>"
    )

    title = "Endpoint identity-gap dashboard"
    lede = (
        f"{n} free-tier API probe{'' if n == 1 else 's'}. "
        "Each endpoint is checked for the contract fields a rollout-marketplace worker would need to publish."
    )
    return page_shell(title, lede, body)


def write_dashboard(dashboard: EndpointDashboard, out_dir: Path | str) -> dict[str, Path]:
    """Persist JSON + HTML side by side under out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "endpoint_dashboard.json"
    html_path = out_path / "endpoint_dashboard.html"
    json_path.write_text(json.dumps(dashboard.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html(dashboard), encoding="utf-8")
    return {"json": json_path, "html": html_path}
