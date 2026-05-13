"""Tests for STEER ``dashboard.device_axis``.

Asserts the optional ``DeviceFingerprint`` field round-trips through
every lab (dense, router, agent), that the env-var helper builds the
fingerprint correctly, that reports without a device bucket as
``L40S (g6e.12xlarge)``, and that dashboards surface a device column /
"Devices observed" header line.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rollout_market.observatory._device import (
    DEFAULT_DEVICE_BUCKET,
    DeviceFingerprint,
    device_bucket_label,
    device_from_env,
)
from rollout_market.observatory.agent_dashboard import (
    build_dashboard as build_agent_dashboard,
    render_html as render_agent_html,
)
from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    TrajectoryDivergenceReport,
    compute_trajectory_divergence,
)
from rollout_market.observatory.dense_dashboard import (
    build_dashboard as build_dense_dashboard,
    render_html as render_dense_html,
)
from rollout_market.observatory.dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
    compute_dense_mismatch,
)
from rollout_market.observatory.router_dashboard import (
    build_dashboard as build_router_dashboard,
    render_html as render_router_html,
)
from rollout_market.observatory.router_mismatch_lab import (
    RouterMismatchReport,
    RouterTrace,
    compute_router_mismatch,
)


def _fp(name: str) -> EngineFingerprint:
    return EngineFingerprint(name=name, version="0.0", fingerprint=f"sha256:{name}")


def _device() -> DeviceFingerprint:
    return DeviceFingerprint(
        vendor="NVIDIA",
        family="H100",
        cuda_compute="9.0",
        vram_gb=80,
        host_label="p5.48xlarge",
    )


# --- DeviceFingerprint + helpers --------------------------------------------


def test_default_bucket_for_missing_device():
    assert device_bucket_label(None) == DEFAULT_DEVICE_BUCKET
    assert "L40S" in DEFAULT_DEVICE_BUCKET


def test_device_bucket_label_includes_host_label():
    d = _device()
    label = device_bucket_label(d)
    assert "H100" in label
    assert "p5.48xlarge" in label


def test_device_from_env_returns_none_when_unset():
    assert device_from_env({}) is None


def test_device_from_env_reads_all_documented_vars():
    env = {
        "DEVICE_VENDOR": "NVIDIA",
        "DEVICE_FAMILY": "H100",
        "DEVICE_CUDA_COMPUTE": "9.0",
        "DEVICE_VRAM_GB": "80",
        "DEVICE_HOST_LABEL": "p5.48xlarge",
    }
    d = device_from_env(env)
    assert d is not None
    assert d.vendor == "NVIDIA"
    assert d.family == "H100"
    assert d.cuda_compute == "9.0"
    assert d.vram_gb == 80
    assert d.host_label == "p5.48xlarge"


def test_device_from_env_with_partial_inputs():
    env = {"DEVICE_FAMILY": "L40S", "DEVICE_HOST_LABEL": "g6e.12xlarge"}
    d = device_from_env(env)
    assert d is not None
    assert d.family == "L40S"
    assert d.host_label == "g6e.12xlarge"
    assert d.cuda_compute is None
    assert d.vram_gb is None


# --- Dense ------------------------------------------------------------------


def test_dense_report_round_trips_with_device(tmp_path: Path):
    rep = compute_dense_mismatch(
        run_id="r0",
        model_id="Qwen/Qwen3-32B",
        checkpoint_digest="sha256:demo",
        tokenizer_hash="tok-x",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        precision_class="bf16",
        prompt_id="p1",
        rollout_logprobs=[-0.5, -0.4],
        trainer_logprobs=[-0.5, -0.4],
        device=_device(),
    )
    assert rep.device == _device()
    out = tmp_path / "rep.json"
    out.write_text(rep.model_dump_json())
    restored = DenseMismatchReport.model_validate_json(out.read_text())
    assert restored.device == _device()


def test_dense_report_defaults_device_to_none():
    rep = compute_dense_mismatch(
        run_id="r0",
        model_id="Qwen/Qwen3-32B",
        checkpoint_digest="sha256:demo",
        tokenizer_hash="tok-x",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        precision_class="bf16",
        prompt_id="p1",
        rollout_logprobs=[-0.5, -0.4],
        trainer_logprobs=[-0.5, -0.4],
    )
    assert rep.device is None
    # Round-trip JSON written before the field existed (no `device` key)
    # still validates because the field is optional.
    payload = json.loads(rep.model_dump_json())
    del payload["device"]
    restored = DenseMismatchReport.model_validate(payload)
    assert restored.device is None


# --- Router -----------------------------------------------------------------


def _trace() -> RouterTrace:
    return RouterTrace(
        num_tokens=2,
        num_layers=2,
        top_k=2,
        num_experts=4,
        expert_ids=[[[0, 1], [1, 2]], [[1, 0], [0, 2]]],
    )


def test_router_report_round_trips_with_device():
    rep = compute_router_mismatch(
        run_id="r0",
        model_id="Qwen3-30B-A3B",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        rollout_trace=_trace(),
        trainer_trace=_trace(),
        device=_device(),
    )
    assert rep.device == _device()
    restored = RouterMismatchReport.model_validate_json(rep.model_dump_json())
    assert restored.device == _device()


def test_router_report_defaults_device_to_none():
    rep = compute_router_mismatch(
        run_id="r0",
        model_id="Qwen3-30B-A3B",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        rollout_trace=_trace(),
        trainer_trace=_trace(),
    )
    assert rep.device is None


# --- Agent ------------------------------------------------------------------


def _trajectory(name: str) -> AgentTrajectory:
    return AgentTrajectory(
        run_id=f"run-{name}",
        task_id="task-1",
        task_text="demo task",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_fp(name),
        available_tools=["search"],
        steps=[
            AgentStep(
                step_idx=0,
                role="assistant",
                content="answer-x",
                finish_reason="stop",
            )
        ],
        success=True,
        terminal_reason="answered",
        total_assistant_tokens=2,
        final_answer="answer-x",
        created_at=datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_agent_report_round_trips_with_device():
    rollout = _trajectory("vllm")
    trainer = _trajectory("fsdp")
    rep = compute_trajectory_divergence(rollout, trainer, device=_device())
    assert rep.device == _device()
    restored = TrajectoryDivergenceReport.model_validate_json(rep.model_dump_json())
    assert restored.device == _device()


def test_agent_report_defaults_device_to_none():
    rep = compute_trajectory_divergence(_trajectory("vllm"), _trajectory("fsdp"))
    assert rep.device is None


# --- Dashboards surface the device pivot ------------------------------------


def _dense_report(*, with_device: bool, idx: int) -> DenseMismatchReport:
    return compute_dense_mismatch(
        run_id=f"r{idx}",
        model_id="Qwen/Qwen3-32B",
        checkpoint_digest="sha256:demo",
        tokenizer_hash="tok-x",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        precision_class="bf16",
        prompt_id="p1",
        rollout_logprobs=[-0.5, -0.4],
        trainer_logprobs=[-0.5, -0.4],
        device=_device() if with_device else None,
    )


def test_dense_dashboard_buckets_missing_device_as_legacy():
    dashboard = build_dense_dashboard([_dense_report(with_device=False, idx=1)])
    page = render_dense_html(dashboard)
    assert "Devices observed" in page
    assert DEFAULT_DEVICE_BUCKET in page
    # Per-run table has a 'device' header column.
    assert "<th>device</th>" in page


def test_dense_dashboard_surfaces_explicit_device():
    dashboard = build_dense_dashboard([_dense_report(with_device=True, idx=1)])
    page = render_dense_html(dashboard)
    assert "Devices observed" in page
    # The H100 family appears in the lede and the per-run table cell.
    assert "H100" in page


def test_router_dashboard_buckets_missing_device():
    rep = compute_router_mismatch(
        run_id="r0",
        model_id="Qwen3-30B-A3B",
        rollout_engine=_fp("vllm"),
        trainer_engine=_fp("fsdp"),
        rollout_trace=_trace(),
        trainer_trace=_trace(),
    )
    dashboard = build_router_dashboard([rep])
    page = render_router_html(dashboard)
    assert "Devices observed" in page
    assert DEFAULT_DEVICE_BUCKET in page
    assert "<th>device</th>" in page


def test_agent_dashboard_viewer_renders_empty_state():
    """The agent page is now a plain trajectory viewer, not a
    comparison dashboard — the device-axis comparison tables are
    gone. With no trajectories on disk the viewer emits the
    empty-state message instead."""
    dashboard = build_agent_dashboard([])
    page = render_agent_html(dashboard, trajectory_dir="/tmp/_nonexistent_agent_dir_")
    assert "Agent trajectory viewer" in page
    assert "Captured trajectories (0)" in page
    # Comparison-era tables are gone.
    assert "<th>device</th>" not in page
    assert "Devices observed" not in page
