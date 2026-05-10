import json
import math
from pathlib import Path

import pytest

from rollout_market.cli import dense_mismatch_lab as cli
from rollout_market.observatory.dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
    compute_dense_mismatch,
    render_html,
    write_dense_mismatch,
)

EXAMPLES = Path(__file__).parents[1] / "examples"


def _engines():
    rollout = EngineFingerprint(name="vllm", version="0.6.0", fingerprint="sha256:rollout")
    trainer = EngineFingerprint(name="transformers", version="4.45.0", fingerprint="sha256:trainer")
    return rollout, trainer


def _baseline_report(**overrides) -> DenseMismatchReport:
    rollout, trainer = _engines()
    kwargs = dict(
        run_id="r-1",
        model_id="model-x",
        checkpoint_digest="sha256:ckpt",
        tokenizer_hash="tok-x",
        rollout_engine=rollout,
        trainer_engine=trainer,
        precision_class="bf16",
        prompt_id="p-1",
        rollout_logprobs=[-0.5, -0.7, -0.3, -0.4],
        trainer_logprobs=[-0.5, -0.7, -0.3, -0.4],
    )
    kwargs.update(overrides)
    return compute_dense_mismatch(**kwargs)


def test_identity_inputs_produce_unit_ess_and_zero_drift():
    report = _baseline_report()
    assert report.num_policy_tokens == 4
    assert report.delta_logprob_mean == 0.0
    assert report.sequence_log_ratio == 0.0
    assert math.isclose(report.ess, 1.0, rel_tol=1e-9)
    assert report.clipped_fraction == 0.0
    assert report.veto_fraction == 0.0
    assert report.max_abs_log_ratio == 0.0


def test_drift_makes_ess_drop_below_one():
    report = _baseline_report(
        rollout_logprobs=[-0.5, -0.7, -0.3, -0.4],
        trainer_logprobs=[-1.5, -2.2, -0.1, 0.0],
    )
    assert report.ess < 1.0
    assert report.delta_logprob_abs_mean > 0.0
    assert report.max_abs_log_ratio > 0.0


def test_mask_filters_tokens():
    report = _baseline_report(
        rollout_logprobs=[-0.5, -0.7, -0.3, -0.4],
        trainer_logprobs=[-0.5, -2.0, -0.3, -0.4],
        mask=[1, 0, 1, 1],
    )
    assert report.num_policy_tokens == 3
    assert report.delta_logprob_mean == 0.0


def test_schema_version_is_pinned():
    report = _baseline_report()
    payload = report.model_dump(mode="json")
    assert payload["schema_version"] == "mismatch_observatory.v0"
    for required in (
        "run_id",
        "model_id",
        "checkpoint_digest",
        "tokenizer_hash",
        "rollout_engine",
        "trainer_engine",
        "precision_class",
        "prompt_id",
        "num_policy_tokens",
        "delta_logprob_mean",
        "delta_logprob_abs_mean",
        "sequence_log_ratio",
        "ess",
        "second_moment",
        "clipped_fraction",
        "veto_fraction",
        "max_abs_log_ratio",
        "top_1pct_gradient_mass",
    ):
        assert required in payload, f"missing acceptance field: {required}"


def test_render_html_is_valid_and_escapes_user_input():
    report = _baseline_report(run_id="<script>alert(1)</script>")
    page = render_html(report)
    assert page.startswith("<!doctype html>")
    assert "</html>" in page.lower()
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "schema_version" in page


def test_render_html_includes_notes_when_present():
    report = _baseline_report(notes=["bf16 vs fp16", "seed=1234"])
    page = render_html(report)
    assert "<h2>Notes</h2>" in page
    assert "bf16 vs fp16" in page


def test_write_dense_mismatch_creates_both_files(tmp_path: Path):
    report = _baseline_report()
    out_dir = tmp_path / "runs" / "20260510T000000Z"
    paths = write_dense_mismatch(report, out_dir)
    assert paths["json"].exists()
    assert paths["html"].exists()
    payload = json.loads(paths["json"].read_text())
    assert payload["schema_version"] == "mismatch_observatory.v0"


def test_compute_rejects_mismatched_lengths():
    rollout, trainer = _engines()
    with pytest.raises(ValueError):
        compute_dense_mismatch(
            run_id="r",
            model_id="m",
            checkpoint_digest="c",
            tokenizer_hash="t",
            rollout_engine=rollout,
            trainer_engine=trainer,
            precision_class="bf16",
            prompt_id="p",
            rollout_logprobs=[-0.1, -0.2],
            trainer_logprobs=[-0.1],
        )


def test_cli_main_round_trips_example_fixture(tmp_path: Path, capsys):
    rc = cli.main(
        [
            "--input",
            str(EXAMPLES / "dense_mismatch_input.json"),
            "--out-root",
            str(tmp_path / "runs"),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2
    json_path = Path(out[0])
    html_path = Path(out[1])
    assert json_path.exists() and json_path.suffix == ".json"
    assert html_path.exists() and html_path.suffix == ".html"
    payload = json.loads(json_path.read_text())
    assert payload["run_id"] == "demo-dense-001"
    assert payload["num_policy_tokens"] == 6
