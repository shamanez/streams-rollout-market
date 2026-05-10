"""Tests for the dashboard launcher's pure functions.

The HTTP server is exercised end to end manually; here we cover the
pieces that map dashboard JSON payloads into one-line summaries and
render the index page.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "serve_dashboards", REPO_ROOT / "scripts" / "live" / "serve_dashboards.py"
)
serve_dashboards = importlib.util.module_from_spec(SPEC)
sys.modules["serve_dashboards"] = serve_dashboards
SPEC.loader.exec_module(serve_dashboards)  # type: ignore[union-attr]


def test_endpoint_summary_format():
    payload = {
        "num_runs": 10,
        "capability_totals": {
            "token_ids_available": 1,
            "sampled_logprobs_available": 3,
            "top_logprobs_available": 3,
            "seed_supported": 1,
        },
    }
    s = serve_dashboards._endpoint_summary(payload)
    assert "10 probes" in s
    assert "sampled_logprobs_available=3/10" in s


def test_dense_summary_with_engine_pairs():
    payload = {
        "engine_pairs": [
            {
                "rollout_engine": "vllm",
                "trainer_engine": "hf-transformers",
                "mean_ess": 0.999,
                "worst_max_abs_log_ratio": 0.165,
            },
        ]
    }
    s = serve_dashboards._dense_summary(payload)
    assert "vllm→hf-transformers" in s
    assert "ESS=0.999" in s


def test_router_summary():
    payload = {
        "engine_pairs": [
            {
                "rollout_engine": "hf-fp8",
                "trainer_engine": "hf-bf16",
                "mean_router_flip_rate": 0.0514,
                "mean_token_expert_disagreement_rate": 0.984,
            }
        ]
    }
    s = serve_dashboards._router_summary(payload)
    assert "hf-fp8→hf-bf16" in s
    assert "set_disagree=0.984" in s


def test_agent_summary_handles_none_first_div_step():
    payload = {
        "engine_pairs": [
            {
                "rollout_engine": "vllm-fp8",
                "trainer_engine": "vllm-bf16",
                "final_answer_match_rate": 0.33,
                "mean_tool_call_jaccard": 0.87,
                "mean_first_divergence_step": None,
            }
        ]
    }
    s = serve_dashboards._agent_summary(payload)
    assert "answer_match=0.33" in s
    assert "first_div_step=0.0" in s


def test_marketplace_summary():
    payload = {
        "jobs_total": 27,
        "submissions": 27,
        "livestore_by_state": {"accepted": 9, "rejected": 12},
    }
    s = serve_dashboards._marketplace_summary(payload)
    assert "jobs=27" in s
    assert "rejected" in s


def test_render_index_marks_missing_cards(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(serve_dashboards, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(serve_dashboards, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(serve_dashboards, "LIVE_ROOT", tmp_path / "runs" / "live")
    page = serve_dashboards.render_index()
    # No dashboards generated → all missing cards.
    assert page.startswith("<!doctype html>")
    assert "missing" in page
    assert "Endpoint identity-gap dashboard" in page
    assert "Agent trajectory dashboard" in page


def test_render_index_marks_ready_when_files_exist(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(serve_dashboards, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(serve_dashboards, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(serve_dashboards, "LIVE_ROOT", tmp_path / "runs" / "live")
    # Create the agent dashboard pair.
    agent_dir = tmp_path / "runs" / "live" / "agent_dashboard"
    agent_dir.mkdir(parents=True)
    (agent_dir / "agent_dashboard.html").write_text("<html>agent</html>")
    (agent_dir / "agent_dashboard.json").write_text(
        '{"engine_pairs":[{"rollout_engine":"x","trainer_engine":"y",'
        '"final_answer_match_rate":0.33,"mean_tool_call_jaccard":0.87,'
        '"mean_first_divergence_step":1.0}]}'
    )
    page = serve_dashboards.render_index()
    assert "agent_dashboard/agent_dashboard.html" in page
    assert "answer_match=0.33" in page


def test_write_index_writes_to_live_root(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(serve_dashboards, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(serve_dashboards, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(serve_dashboards, "LIVE_ROOT", tmp_path / "runs" / "live")
    target = serve_dashboards.write_index()
    assert target.exists()
    assert target.name == "index.html"
    assert target.parent.name == "live"
