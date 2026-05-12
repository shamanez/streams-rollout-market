"""Tests for the per-(pair, task) click-through pages published by
scripts/live/publish_dashboards.py::_copy_agent_diff_click_through.

The pair script writes
  runs/live/agent_diff/<pair-slug>/<ts-uuid>/agent_divergence_report.{json,html}
With replicates there are multiple report timestamps per (pair, task);
the publish step must pick the latest per (pair, task) and copy it to
  docs/agent_diff/<pair-slug>/<task-id>.html
so the public site has a stable URL per task.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "live" / "publish_dashboards.py"
_spec = importlib.util.spec_from_file_location("publish_dashboards", _SCRIPT)
publish_dashboards = importlib.util.module_from_spec(_spec)
sys.modules["publish_dashboards"] = publish_dashboards
_spec.loader.exec_module(publish_dashboards)  # type: ignore[union-attr]


def _write_report(
    src_root: Path,
    pair_slug: str,
    task_id: str,
    *,
    ts_uuid: str,
    html_body: str,
    mtime: float | None = None,
) -> Path:
    """Write a fake (pair, run) report pair under src_root."""
    run_dir = src_root / pair_slug / ts_uuid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "agent_divergence_report.json").write_text(
        json.dumps({"task_id": task_id, "schema_version": "agent_divergence.v0"}),
        encoding="utf-8",
    )
    html_path = run_dir / "agent_divergence_report.html"
    html_path.write_text(html_body, encoding="utf-8")
    if mtime is not None:
        os.utime(run_dir, (mtime, mtime))
    return html_path


def test_copy_clickthrough_writes_one_page_per_pair_task(
    tmp_path: Path, monkeypatch
) -> None:
    """Each (pair_slug, task_id) maps to exactly one HTML page in out_dir."""
    src_root = tmp_path / "runs_live"
    _write_report(
        src_root / "agent_diff",
        "hermes-qwen3-32b-fp8-vs-bf16",
        "code-fix-factorial",
        ts_uuid="20260512T000000Z-aaaa",
        html_body="<h1>code-fix-factorial fp8 vs bf16</h1>",
    )
    _write_report(
        src_root / "agent_diff",
        "hermes-qwen3-32b-bf16_seedB-vs-bf16",
        "code-fix-factorial",
        ts_uuid="20260512T000000Z-bbbb",
        html_body="<h1>code-fix-factorial seedB vs bf16</h1>",
    )
    monkeypatch.setattr(publish_dashboards, "LIVE_ROOT", src_root)
    out_dir = tmp_path / "docs"
    n = publish_dashboards._copy_agent_diff_click_through(out_dir)
    assert n == 2
    fp8_page = out_dir / "agent_diff" / "hermes-qwen3-32b-fp8-vs-bf16" / "code-fix-factorial.html"
    seedB_page = out_dir / "agent_diff" / "hermes-qwen3-32b-bf16_seedB-vs-bf16" / "code-fix-factorial.html"
    assert fp8_page.exists()
    assert seedB_page.exists()
    assert "fp8 vs bf16" in fp8_page.read_text(encoding="utf-8")
    assert "seedB vs bf16" in seedB_page.read_text(encoding="utf-8")


def test_copy_clickthrough_picks_newest_replicate(
    tmp_path: Path, monkeypatch
) -> None:
    """With multiple replicate runs per (pair, task), the latest-mtime
    report wins."""
    src_root = tmp_path / "runs_live"
    # Older first
    _write_report(
        src_root / "agent_diff",
        "hermes-qwen3-32b-fp8-vs-bf16",
        "code-fix-factorial",
        ts_uuid="20260512T000000Z-oldd",
        html_body="<h1>OLD</h1>",
        mtime=time.time() - 1000,
    )
    _write_report(
        src_root / "agent_diff",
        "hermes-qwen3-32b-fp8-vs-bf16",
        "code-fix-factorial",
        ts_uuid="20260512T010000Z-neww",
        html_body="<h1>NEW</h1>",
        mtime=time.time(),
    )
    monkeypatch.setattr(publish_dashboards, "LIVE_ROOT", src_root)
    out_dir = tmp_path / "docs"
    publish_dashboards._copy_agent_diff_click_through(out_dir)
    page = (
        out_dir / "agent_diff" / "hermes-qwen3-32b-fp8-vs-bf16" / "code-fix-factorial.html"
    )
    assert "NEW" in page.read_text(encoding="utf-8")
    assert "OLD" not in page.read_text(encoding="utf-8")


def test_copy_clickthrough_returns_zero_when_no_reports(
    tmp_path: Path, monkeypatch
) -> None:
    """Missing or empty runs/live/agent_diff/ -> 0 pages, no error."""
    monkeypatch.setattr(publish_dashboards, "LIVE_ROOT", tmp_path / "empty")
    n = publish_dashboards._copy_agent_diff_click_through(tmp_path / "docs")
    assert n == 0


def test_copy_clickthrough_sanitises_task_id_for_filesystem(
    tmp_path: Path, monkeypatch
) -> None:
    """Task IDs containing slashes / spaces / special chars must be
    sanitized to a safe filename."""
    src_root = tmp_path / "runs_live"
    _write_report(
        src_root / "agent_diff",
        "hermes-test",
        "task/with slashes & weird?chars",
        ts_uuid="20260512T000000Z-aaaa",
        html_body="<h1>x</h1>",
    )
    monkeypatch.setattr(publish_dashboards, "LIVE_ROOT", src_root)
    out_dir = tmp_path / "docs"
    publish_dashboards._copy_agent_diff_click_through(out_dir)
    # Slashes -> underscores; & -> underscore; ? -> underscore; space -> underscore.
    page_dir = out_dir / "agent_diff" / "hermes-test"
    assert page_dir.exists()
    files = list(page_dir.glob("*.html"))
    assert len(files) == 1
    assert "/" not in files[0].name
    assert " " not in files[0].name


def test_copy_clickthrough_skips_reports_with_missing_html(
    tmp_path: Path, monkeypatch
) -> None:
    """JSON without a sibling HTML must not crash the publish step."""
    src_root = tmp_path / "runs_live"
    run_dir = src_root / "agent_diff" / "hermes-test" / "20260512T000000Z-aaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "agent_divergence_report.json").write_text(
        json.dumps({"task_id": "t", "schema_version": "agent_divergence.v0"}),
        encoding="utf-8",
    )
    # Intentionally no .html written.
    monkeypatch.setattr(publish_dashboards, "LIVE_ROOT", src_root)
    n = publish_dashboards._copy_agent_diff_click_through(tmp_path / "docs")
    assert n == 0
