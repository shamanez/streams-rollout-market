"""Small MoE router mismatch lab.

Compares per-token, per-layer expert routing decisions between a rollout
backend and a reference forward pass for the same checkpoint and tokens.
The lab produces a stable report with two headline metrics:

- ``router_flip_rate``: fraction of (token, layer) pairs whose top-1 expert
  differs between rollout and trainer.
- ``token_expert_disagreement_rate``: fraction of tokens for which the
  *set* of routed experts (top-k, order-insensitive) differs in at least
  one layer.

Like the dense lab, this module is metrics-only and trainer-free. Capturing
the routing traces is out of scope here; the contract simply requires that
both sides be aligned by token and layer.
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ._device import DeviceFingerprint
from .dense_mismatch_lab import EngineFingerprint


class RouterTrace(BaseModel):
    """Routing decisions for one trajectory under one engine.

    ``expert_ids`` is shaped [num_tokens][num_layers][top_k]. All three axes
    must be consistent across tokens and layers — ragged traces are not
    supported by this contract.
    """

    schema_version: Literal["router_trace.v0"] = "router_trace.v0"
    num_layers: int
    num_experts: int
    top_k: int
    expert_ids: list[list[list[int]]]

    @model_validator(mode="after")
    def _validate_shape(self) -> "RouterTrace":
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        for t_idx, per_token in enumerate(self.expert_ids):
            if len(per_token) != self.num_layers:
                raise ValueError(
                    f"token {t_idx} has {len(per_token)} layers, expected {self.num_layers}"
                )
            for l_idx, per_layer in enumerate(per_token):
                if len(per_layer) != self.top_k:
                    raise ValueError(
                        f"token {t_idx} layer {l_idx} has {len(per_layer)} experts, "
                        f"expected top_k={self.top_k}"
                    )
                for expert in per_layer:
                    if expert < 0 or expert >= self.num_experts:
                        raise ValueError(
                            f"expert id {expert} out of range [0, {self.num_experts})"
                        )
        return self

    @property
    def num_tokens(self) -> int:
        return len(self.expert_ids)


class RouterMismatchReport(BaseModel):
    """Stable JSON contract for one router-mismatch comparison."""

    schema_version: Literal["router_mismatch.v0"] = "router_mismatch.v0"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_id: str
    rollout_engine: EngineFingerprint
    trainer_engine: EngineFingerprint

    num_tokens: int
    num_layers: int
    top_k: int
    num_experts: int

    router_flip_rate: float
    token_expert_disagreement_rate: float
    layer_flip_rates: list[float]

    device: DeviceFingerprint | None = None

    notes: list[str] = Field(default_factory=list)


def compute_router_mismatch(
    *,
    run_id: str,
    model_id: str,
    rollout_engine: EngineFingerprint,
    trainer_engine: EngineFingerprint,
    rollout_trace: RouterTrace,
    trainer_trace: RouterTrace,
    notes: list[str] | None = None,
    device: DeviceFingerprint | None = None,
) -> RouterMismatchReport:
    """Compute router-mismatch metrics from two aligned traces.

    Both traces must have the same num_tokens, num_layers, num_experts, and
    top_k. Token and layer indices are positional and must align.
    """
    for field in ("num_layers", "num_experts", "top_k"):
        a = getattr(rollout_trace, field)
        b = getattr(trainer_trace, field)
        if a != b:
            raise ValueError(f"trace mismatch on {field}: rollout={a}, trainer={b}")
    if rollout_trace.num_tokens != trainer_trace.num_tokens:
        raise ValueError(
            f"trace mismatch on num_tokens: rollout={rollout_trace.num_tokens}, "
            f"trainer={trainer_trace.num_tokens}"
        )

    num_tokens = rollout_trace.num_tokens
    num_layers = rollout_trace.num_layers
    if num_tokens == 0:
        raise ValueError("router mismatch requires at least one token")

    layer_flip_counts = [0] * num_layers
    token_disagreement_count = 0
    total_pairs = num_tokens * num_layers

    for t_idx in range(num_tokens):
        token_disagrees = False
        for l_idx in range(num_layers):
            roll = rollout_trace.expert_ids[t_idx][l_idx]
            train = trainer_trace.expert_ids[t_idx][l_idx]
            if roll[0] != train[0]:
                layer_flip_counts[l_idx] += 1
            if set(roll) != set(train):
                token_disagrees = True
        if token_disagrees:
            token_disagreement_count += 1

    router_flip_rate = sum(layer_flip_counts) / total_pairs
    token_expert_disagreement_rate = token_disagreement_count / num_tokens
    layer_flip_rates = [count / num_tokens for count in layer_flip_counts]

    return RouterMismatchReport(
        run_id=run_id,
        model_id=model_id,
        rollout_engine=rollout_engine,
        trainer_engine=trainer_engine,
        num_tokens=num_tokens,
        num_layers=num_layers,
        top_k=rollout_trace.top_k,
        num_experts=rollout_trace.num_experts,
        router_flip_rate=router_flip_rate,
        token_expert_disagreement_rate=token_expert_disagreement_rate,
        layer_flip_rates=layer_flip_rates,
        device=device,
        notes=list(notes or []),
    )


def _row(label: str, value: object) -> str:
    rendered = format(value) if not isinstance(value, str) else value
    return f"<tr><th>{_html.escape(label)}</th><td>{_html.escape(rendered)}</td></tr>"


def render_html(report: RouterMismatchReport) -> str:
    """Render a self-contained HTML view of one router mismatch report."""
    summary_rows = "".join(
        [
            _row("schema_version", report.schema_version),
            _row("run_id", report.run_id),
            _row("created_at", report.created_at.isoformat()),
            _row("model_id", report.model_id),
            _row(
                "rollout_engine",
                f"{report.rollout_engine.name} {report.rollout_engine.version} "
                f"({report.rollout_engine.fingerprint})",
            ),
            _row(
                "trainer_engine",
                f"{report.trainer_engine.name} {report.trainer_engine.version} "
                f"({report.trainer_engine.fingerprint})",
            ),
            _row("num_tokens", report.num_tokens),
            _row("num_layers", report.num_layers),
            _row("top_k", report.top_k),
            _row("num_experts", report.num_experts),
            _row("router_flip_rate", f"{report.router_flip_rate:.6f}"),
            _row(
                "token_expert_disagreement_rate",
                f"{report.token_expert_disagreement_rate:.6f}",
            ),
        ]
    )
    layer_rows = "".join(
        f"<tr><td>{idx}</td><td>{rate:.6f}</td></tr>"
        for idx, rate in enumerate(report.layer_flip_rates)
    )
    notes_html = ""
    if report.notes:
        items = "".join(f"<li>{_html.escape(n)}</li>" for n in report.notes)
        notes_html = f"<h2>Notes</h2><ul>{items}</ul>"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Router mismatch — {_html.escape(report.run_id)}</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;color:#111}"
        "table{border-collapse:collapse;width:100%;margin-bottom:1rem}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5;width:40%}"
        "h1{font-size:1.2rem}"
        "h2{font-size:1rem}"
        "</style></head><body>"
        f"<h1>Router mismatch — {_html.escape(report.run_id)}</h1>"
        f"<table>{summary_rows}</table>"
        "<h2>Per-layer top-1 flip rate</h2>"
        f"<table><thead><tr><th>layer</th><th>flip_rate</th></tr></thead>"
        f"<tbody>{layer_rows}</tbody></table>"
        f"{notes_html}"
        "</body></html>"
    )


def write_router_mismatch(report: RouterMismatchReport, out_dir: Path | str) -> dict[str, Path]:
    """Write JSON and HTML side by side under out_dir. Returns the paths written."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "router_mismatch_report.json"
    html_path = out_path / "router_mismatch_report.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def load_input_fixture(path: Path | str) -> dict:
    """Load a CLI fixture: a JSON file mirroring compute_router_mismatch kwargs."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
