import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollout_market.cli import router_mismatch_lab as cli
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint
from rollout_market.observatory.router_mismatch_lab import (
    RouterMismatchReport,
    RouterTrace,
    compute_router_mismatch,
    render_html,
    write_router_mismatch,
)

EXAMPLES = Path(__file__).parents[1] / "examples"


def _engines():
    rollout = EngineFingerprint(name="vllm", version="0.6.0", fingerprint="sha256:rollout")
    trainer = EngineFingerprint(name="transformers", version="4.45.0", fingerprint="sha256:trainer")
    return rollout, trainer


def _trace(expert_ids: list[list[list[int]]], num_experts: int = 8, top_k: int = 2) -> RouterTrace:
    num_layers = len(expert_ids[0]) if expert_ids else 1
    return RouterTrace(
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        expert_ids=expert_ids,
    )


def _compute(rollout: RouterTrace, trainer: RouterTrace, **overrides) -> RouterMismatchReport:
    rollout_engine, trainer_engine = _engines()
    kwargs = dict(
        run_id="r-1",
        model_id="moe-x",
        rollout_engine=rollout_engine,
        trainer_engine=trainer_engine,
        rollout_trace=rollout,
        trainer_trace=trainer,
    )
    kwargs.update(overrides)
    return compute_router_mismatch(**kwargs)


def test_identical_traces_have_zero_mismatch():
    trace = _trace([[[0, 1], [3, 4]], [[2, 5], [6, 7]]])
    other = _trace([[[0, 1], [3, 4]], [[2, 5], [6, 7]]])
    report = _compute(trace, other)
    assert report.router_flip_rate == 0.0
    assert report.token_expert_disagreement_rate == 0.0
    assert report.layer_flip_rates == [0.0, 0.0]


def test_top1_flip_counted_per_layer():
    rollout = _trace([[[0, 1], [3, 4]]])  # 1 token, 2 layers
    trainer = _trace([[[5, 1], [3, 4]]])  # layer 0 top-1 flip
    report = _compute(rollout, trainer)
    assert report.layer_flip_rates == [1.0, 0.0]
    assert report.router_flip_rate == 0.5


def test_top1_match_but_set_disagrees_marks_token():
    rollout = _trace([[[0, 1], [3, 4]]])
    trainer = _trace([[[0, 7], [3, 4]]])  # top-1 same, but {0,7} != {0,1}
    report = _compute(rollout, trainer)
    assert report.router_flip_rate == 0.0
    assert report.token_expert_disagreement_rate == 1.0


def test_top1_set_order_difference_does_not_disagree():
    rollout = _trace([[[0, 1], [3, 4]]])
    trainer = _trace([[[1, 0], [4, 3]]])  # same sets, order swapped, top-1 differs
    report = _compute(rollout, trainer)
    assert report.router_flip_rate == 1.0
    assert report.token_expert_disagreement_rate == 0.0


def test_token_disagreement_rate_counts_unique_tokens():
    rollout = _trace([
        [[0, 1], [3, 4]],
        [[0, 1], [3, 4]],
        [[0, 1], [3, 4]],
    ])
    trainer = _trace([
        [[0, 1], [3, 4]],
        [[0, 7], [3, 4]],  # disagreement on token 1
        [[0, 1], [3, 4]],
    ])
    report = _compute(rollout, trainer)
    assert report.token_expert_disagreement_rate == pytest.approx(1 / 3)


def test_router_trace_validates_shape():
    with pytest.raises(ValidationError):
        RouterTrace(num_layers=2, num_experts=4, top_k=2, expert_ids=[[[0, 1]]])  # only 1 layer


def test_router_trace_rejects_top_k_gt_experts():
    with pytest.raises(ValidationError):
        RouterTrace(num_layers=1, num_experts=2, top_k=3, expert_ids=[[[0, 1, 0]]])


def test_router_trace_rejects_out_of_range_expert_id():
    with pytest.raises(ValidationError):
        RouterTrace(num_layers=1, num_experts=4, top_k=2, expert_ids=[[[0, 99]]])


def test_compute_rejects_size_mismatch():
    rollout = _trace([[[0, 1], [3, 4]]])
    trainer = _trace([[[0, 1], [3, 4]], [[0, 1], [3, 4]]])
    with pytest.raises(ValueError):
        _compute(rollout, trainer)


def test_compute_rejects_topk_mismatch():
    rollout = RouterTrace(num_layers=1, num_experts=4, top_k=2, expert_ids=[[[0, 1]]])
    trainer = RouterTrace(num_layers=1, num_experts=4, top_k=1, expert_ids=[[[0]]])
    with pytest.raises(ValueError):
        _compute(rollout, trainer)


def test_compute_rejects_zero_tokens():
    rollout = _trace([])
    trainer = _trace([])
    with pytest.raises(ValueError):
        _compute(rollout, trainer)


def test_render_html_escapes_user_input():
    rollout = _trace([[[0, 1], [3, 4]]])
    trainer = _trace([[[0, 1], [3, 4]]])
    report = _compute(rollout, trainer, run_id="<img src=x>")
    page = render_html(report)
    assert "<img src=x>" not in page
    assert "&lt;img src=x&gt;" in page
    assert "Per-layer top-1 flip rate" in page


def test_write_router_mismatch_creates_both_files(tmp_path: Path):
    rollout = _trace([[[0, 1], [3, 4]]])
    trainer = _trace([[[0, 1], [3, 4]]])
    report = _compute(rollout, trainer)
    paths = write_router_mismatch(report, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["html"].exists()
    payload = json.loads(paths["json"].read_text())
    assert payload["schema_version"] == "router_mismatch.v0"


def test_cli_main_round_trips_example_fixture(tmp_path: Path, capsys):
    rc = cli.main(
        [
            "--input",
            str(EXAMPLES / "router_mismatch_input.json"),
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
    assert payload["run_id"] == "demo-router-001"
    assert payload["num_tokens"] == 3
    assert payload["num_layers"] == 2
    assert "router_flip_rate" in payload
    assert "token_expert_disagreement_rate" in payload


def test_acceptance_fields_present():
    rollout = _trace([[[0, 1], [3, 4]]])
    trainer = _trace([[[0, 1], [3, 4]]])
    report = _compute(rollout, trainer)
    payload = report.model_dump(mode="json")
    for required in (
        "router_flip_rate",
        "token_expert_disagreement_rate",
        "layer_flip_rates",
        "rollout_engine",
        "trainer_engine",
    ):
        assert required in payload
