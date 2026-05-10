from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rollout_market.feedback import FeedbackAggregator, QualityStats, TrainerFeedback


def _fb(**overrides) -> TrainerFeedback:
    base = dict(
        group_id="g1",
        worker_id="w0",
        engine_name="vllm",
        engine_version="0.6.0",
        policy_version="v0001",
        num_policy_tokens=12,
        ess_observed=0.9,
        clipped_fraction_observed=0.05,
        accepted_by_trainer=True,
        current_policy_logprob_mean=-0.4,
    )
    base.update(overrides)
    return TrainerFeedback.model_validate(base)


def test_feedback_validates_required_fields():
    fb = _fb()
    assert fb.group_id == "g1"
    assert fb.worker_id == "w0"
    assert fb.engine_name == "vllm"


def test_feedback_rejects_blank_attribution():
    with pytest.raises(ValidationError):
        _fb(worker_id="")


def test_feedback_rejects_out_of_range_ess():
    with pytest.raises(ValidationError):
        _fb(ess_observed=1.5)
    with pytest.raises(ValidationError):
        _fb(ess_observed=-0.1)


def test_feedback_rejects_out_of_range_clipped():
    with pytest.raises(ValidationError):
        _fb(clipped_fraction_observed=1.2)


def test_feedback_rejects_negative_token_count():
    with pytest.raises(ValidationError):
        _fb(num_policy_tokens=-1)


def test_feedback_rejects_rejection_reason_when_accepted():
    with pytest.raises(ValidationError):
        _fb(accepted_by_trainer=True, trainer_rejection_reason="low_ess")


def test_aggregator_ingest_counts_and_acceptance_rate():
    agg = FeedbackAggregator()
    agg.ingest(_fb(group_id="g1", accepted_by_trainer=True, ess_observed=0.9))
    agg.ingest(
        _fb(
            group_id="g2",
            accepted_by_trainer=False,
            trainer_rejection_reason="low_observed_ess",
            ess_observed=0.4,
        )
    )
    stats: QualityStats = agg.stats_for_worker("w0")
    assert stats.count == 2
    assert stats.accepted_count == 1
    assert stats.acceptance_rate == 0.5
    assert stats.mean_ess == pytest.approx(0.65)
    assert stats.rejection_reasons == {"low_observed_ess": 1}


def test_aggregator_separates_workers():
    agg = FeedbackAggregator()
    agg.ingest(_fb(group_id="g1", worker_id="wa"))
    agg.ingest(_fb(group_id="g2", worker_id="wb"))
    assert agg.stats_for_worker("wa").count == 1
    assert agg.stats_for_worker("wb").count == 1
    assert set(agg.all_worker_stats()) == {"wa", "wb"}


def test_aggregator_separates_engines():
    agg = FeedbackAggregator()
    agg.ingest(_fb(group_id="g1", engine_name="vllm"))
    agg.ingest(_fb(group_id="g2", engine_name="sglang"))
    assert agg.stats_for_engine("vllm").count == 1
    assert agg.stats_for_engine("sglang").count == 1


def test_aggregator_separates_policies():
    agg = FeedbackAggregator()
    agg.ingest(_fb(group_id="g1", policy_version="v1"))
    agg.ingest(_fb(group_id="g2", policy_version="v2"))
    assert agg.stats_for_policy("v1").count == 1
    assert agg.stats_for_policy("v2").count == 1


def test_aggregator_is_idempotent_on_group_id():
    agg = FeedbackAggregator()
    agg.ingest(_fb(group_id="g1", ess_observed=0.9))
    agg.ingest(_fb(group_id="g1", ess_observed=0.1))
    assert agg.stats_for_worker("w0").count == 1
    assert agg.stats_for_worker("w0").mean_ess == pytest.approx(0.9)


def test_aggregator_records_last_reported_at():
    agg = FeedbackAggregator()
    t = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    agg.ingest(_fb(group_id="g1", reported_at=t))
    assert agg.stats_for_worker("w0").last_reported_at == t


def test_unknown_keys_return_empty_stats():
    agg = FeedbackAggregator()
    stats = agg.stats_for_worker("nope")
    assert stats.count == 0
    assert stats.acceptance_rate == 0.0
    assert stats.mean_ess == 0.0


def test_quality_stats_as_dict_round_trips():
    agg = FeedbackAggregator()
    agg.ingest(_fb(group_id="g1", ess_observed=0.8, clipped_fraction_observed=0.1))
    payload = agg.stats_for_worker("w0").as_dict()
    for required in (
        "count",
        "accepted_count",
        "acceptance_rate",
        "mean_ess",
        "mean_clipped_fraction",
        "mean_current_policy_logprob",
        "sum_num_policy_tokens",
        "last_reported_at",
        "rejection_reasons",
    ):
        assert required in payload


def test_aggregator_ingest_all_handles_iterable():
    agg = FeedbackAggregator()
    agg.ingest_all([_fb(group_id="g1"), _fb(group_id="g2"), _fb(group_id="g1")])
    assert len(agg) == 2


def test_aggregator_does_not_mutate_trainer_logic():
    """The aggregator only reads from feedback; it must not call back."""
    agg = FeedbackAggregator()
    fb = _fb(group_id="g1")
    agg.ingest(fb)
    # Ingestion must not have mutated the input record (Pydantic models are
    # frozen-ish at the field level, but exercise a snapshot here).
    snapshot = fb.model_dump()
    assert snapshot["group_id"] == "g1"
    assert snapshot["accepted_by_trainer"] is True
