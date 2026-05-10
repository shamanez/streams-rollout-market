"""Controlled dense mismatch lab.

Computes the mismatch-observatory metrics contract for a single
(rollout_logprobs, trainer_logprobs) pair from one checkpoint served by two
backends. Produces a stable JSON contract and a small standalone HTML view.

The lab is metrics-only and trainer-free: it does not perform any gradient
step, optimizer update, or reward-model evaluation. All math lives in
``mismatch_metrics.summarize_logprob_mismatch`` — this module is a contract
shell + renderer around it.
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..mismatch_metrics import summarize_logprob_mismatch


class EngineFingerprint(BaseModel):
    name: str
    version: str
    fingerprint: str


class DenseMismatchReport(BaseModel):
    """Stable JSON contract for one dense mismatch comparison.

    Mirrors experiments/mismatch_observatory/metrics_contract.json so a
    downstream dashboard can render runs without any per-experiment glue.
    """

    schema_version: Literal["mismatch_observatory.v0"] = "mismatch_observatory.v0"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_id: str
    checkpoint_digest: str
    tokenizer_hash: str
    rollout_engine: EngineFingerprint
    trainer_engine: EngineFingerprint
    precision_class: str
    quantization_class: str | None = None
    prompt_id: str

    num_policy_tokens: int
    delta_logprob_mean: float
    delta_logprob_abs_mean: float
    sequence_log_ratio: float
    ess: float
    second_moment: float
    clipped_fraction: float
    veto_fraction: float
    max_abs_log_ratio: float
    top_1pct_gradient_mass: float

    notes: list[str] = Field(default_factory=list)


def compute_dense_mismatch(
    *,
    run_id: str,
    model_id: str,
    checkpoint_digest: str,
    tokenizer_hash: str,
    rollout_engine: EngineFingerprint,
    trainer_engine: EngineFingerprint,
    precision_class: str,
    prompt_id: str,
    rollout_logprobs: list[float],
    trainer_logprobs: list[float],
    mask: list[int] | None = None,
    quantization_class: str | None = None,
    clamp: float = 20.0,
    veto_abs_log_ratio: float = 30.0,
    notes: list[str] | None = None,
) -> DenseMismatchReport:
    """Compute the dense mismatch report for one prompt under two backends.

    rollout_logprobs are sampled-token log probabilities under the serving
    backend; trainer_logprobs are the same token IDs scored by the reference
    forward pass. Token alignment is the caller's responsibility — this
    function only enforces equal lengths.
    """
    summary = summarize_logprob_mismatch(
        rollout_logprobs,
        trainer_logprobs,
        mask=mask,
        clamp=clamp,
        veto_abs_log_ratio=veto_abs_log_ratio,
    )
    return DenseMismatchReport(
        run_id=run_id,
        model_id=model_id,
        checkpoint_digest=checkpoint_digest,
        tokenizer_hash=tokenizer_hash,
        rollout_engine=rollout_engine,
        trainer_engine=trainer_engine,
        precision_class=precision_class,
        quantization_class=quantization_class,
        prompt_id=prompt_id,
        num_policy_tokens=summary.num_tokens,
        delta_logprob_mean=summary.delta_logprob_mean,
        delta_logprob_abs_mean=summary.delta_logprob_abs_mean,
        sequence_log_ratio=summary.sequence_log_ratio,
        ess=summary.ess,
        second_moment=summary.second_moment,
        clipped_fraction=summary.clipped_fraction,
        veto_fraction=summary.veto_fraction,
        max_abs_log_ratio=summary.max_abs_log_ratio,
        top_1pct_gradient_mass=summary.top_1pct_gradient_mass,
        notes=list(notes or []),
    )


def _row(label: str, value: object) -> str:
    return (
        f"<tr><th>{_html.escape(label)}</th>"
        f"<td>{_html.escape(format(value) if not isinstance(value, str) else value)}</td></tr>"
    )


def render_html(report: DenseMismatchReport) -> str:
    """Render a self-contained, dependency-free HTML page for one report.

    Output is plain HTML5 with inline CSS so the file works when emailed,
    served as a static asset, or opened from disk without a build step.
    All user-controlled fields are escaped.
    """
    rows = "".join(
        [
            _row("schema_version", report.schema_version),
            _row("run_id", report.run_id),
            _row("created_at", report.created_at.isoformat()),
            _row("model_id", report.model_id),
            _row("checkpoint_digest", report.checkpoint_digest),
            _row("tokenizer_hash", report.tokenizer_hash),
            _row("precision_class", report.precision_class),
            _row("quantization_class", report.quantization_class or "—"),
            _row("prompt_id", report.prompt_id),
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
            _row("num_policy_tokens", report.num_policy_tokens),
            _row("delta_logprob_mean", f"{report.delta_logprob_mean:.6f}"),
            _row("delta_logprob_abs_mean", f"{report.delta_logprob_abs_mean:.6f}"),
            _row("sequence_log_ratio", f"{report.sequence_log_ratio:.6f}"),
            _row("ess", f"{report.ess:.6f}"),
            _row("second_moment", f"{report.second_moment:.6f}"),
            _row("clipped_fraction", f"{report.clipped_fraction:.6f}"),
            _row("veto_fraction", f"{report.veto_fraction:.6f}"),
            _row("max_abs_log_ratio", f"{report.max_abs_log_ratio:.6f}"),
            _row("top_1pct_gradient_mass", f"{report.top_1pct_gradient_mass:.6f}"),
        ]
    )
    notes_html = ""
    if report.notes:
        items = "".join(f"<li>{_html.escape(n)}</li>" for n in report.notes)
        notes_html = f"<h2>Notes</h2><ul>{items}</ul>"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Dense mismatch — {_html.escape(report.run_id)}</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;color:#111}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;vertical-align:top}"
        "th{background:#f5f5f5;width:40%}"
        "h1{font-size:1.2rem}"
        "code{font-family:ui-monospace,monospace}"
        "</style></head><body>"
        f"<h1>Dense mismatch report — {_html.escape(report.run_id)}</h1>"
        f"<table>{rows}</table>"
        f"{notes_html}"
        "</body></html>"
    )


def write_dense_mismatch(report: DenseMismatchReport, out_dir: Path | str) -> dict[str, Path]:
    """Write JSON and HTML side by side under out_dir. Returns the paths written."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "dense_mismatch_report.json"
    html_path = out_path / "dense_mismatch_report.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def load_input_fixture(path: Path | str) -> dict:
    """Load a CLI input fixture: a JSON file with all compute_dense_mismatch kwargs.

    The fixture format mirrors the function signature so a new run is one file.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
