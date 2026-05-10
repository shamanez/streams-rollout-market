from rollout_market.mismatch_metrics import summarize_logprob_mismatch


def test_summarize_logprob_mismatch_accepts_mask():
    summary = summarize_logprob_mismatch(
        rollout_logprobs=[-1.0, -2.0, -3.0],
        trainer_logprobs=[-1.1, -1.7, -3.0],
        mask=[1, 1, 0],
    )
    assert summary.num_tokens == 2
    assert summary.ess > 0
    assert summary.clipped_fraction == 0


def test_summarize_logprob_mismatch_rejects_empty_mask():
    try:
        summarize_logprob_mismatch([-1.0], [-1.0], mask=[0])
    except ValueError as exc:
        assert "at least one token" in str(exc)
    else:
        raise AssertionError("expected ValueError")
