"""Tests for ``scripts/live/remote_capture_driver.py``.

The driver's job is to put the chat completion request through the
SAME ``inject_logprobs`` augmentation that ``logprob_capture_proxy``
uses, then push the raw response through the SAME ``extract_probes``
contract. This test asserts the driver's per-turn sidecar record is
field-for-field equivalent to what the proxy emits — so the
downstream ``merge_probes_into_trajectories.py`` stays a no-op.

We don't actually POST anything; we monkey-patch ``requests.post``
to return a fixture chat-completion response.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/live/ is not a package — wire it onto sys.path so we can
# import the driver and the proxy helpers directly.
SCRIPTS_LIVE = Path(__file__).resolve().parent.parent / "scripts" / "live"
sys.path.insert(0, str(SCRIPTS_LIVE))


@pytest.fixture
def fake_response_body() -> bytes:
    """Minimal vLLM chat-completions response with all four probe
    fields populated. Mirrors what the patched dev wheel emits for
    a tiny single-token generation.
    """
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "Qwen/Qwen3-30B-A3B",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "4"},
                "logprobs": {
                    "content": [
                        {"token": "4", "logprob": -0.1, "token_id": 19},
                        {"token": "\n", "logprob": -0.2, "token_id": 198},
                    ]
                },
                "token_ids": [19, 198],
                # Shape [gen_tokens, 48_layers, 8_top_k] truncated for the
                # test — the proxy uses usage.completion_tokens to trim a
                # combined prefill+decode buffer down to the gen tail.
                "routed_experts": [
                    [[1, 2, 3, 4, 5, 6, 7, 8]] * 48,  # token 0
                    [[8, 7, 6, 5, 4, 3, 2, 1]] * 48,  # token 1
                ],
            }
        ],
        "prompt_token_ids": [101, 102, 103],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    return json.dumps(payload).encode("utf-8")


def test_driver_captures_same_fields_as_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_response_body: bytes,
) -> None:
    from logprob_capture_proxy import extract_probes, inject_logprobs

    import remote_capture_driver as driver

    sidecar = tmp_path / "probes.jsonl"

    # Capture what the driver POSTs so we can prove augmentation
    # matches inject_logprobs.
    captured: dict = {}

    class _FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return json.loads(self.content.decode("utf-8"))

    def fake_post(url: str, *, data: bytes, headers: dict, timeout: float) -> _FakeResponse:
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        return _FakeResponse(fake_response_body)

    monkeypatch.setattr(driver.requests, "post", fake_post)

    rsp = driver._chat_once_with_capture(
        base_url="http://localhost:8000",
        model="Qwen/Qwen3-30B-A3B",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=8,
        timeout=10.0,
        sidecar_path=sidecar,
        prompt_index=0,
    )

    # 1. The driver's request body must match what inject_logprobs would produce.
    raw = json.dumps(
        {
            "model": "Qwen/Qwen3-30B-A3B",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "tools": driver.TOOLS_OPENAI_SPEC,
            "tool_choice": "auto",
            "max_tokens": 8,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    expected_augmented = inject_logprobs(raw)
    assert captured["data"] == expected_augmented, (
        "driver request body must match inject_logprobs output exactly"
    )

    # 2. Sidecar file: one line per turn, parses as JSON.
    assert sidecar.exists(), "driver must write the sidecar JSONL"
    lines = sidecar.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    driver_record = json.loads(lines[0])

    # 3. Driver record must match what extract_probes would have
    #    emitted for the same (request, response) pair — modulo the
    #    timestamp the driver adds.
    expected_record = extract_probes(
        expected_augmented,
        fake_response_body,
        prompt_index=0,
        turn_idx=0,
    )
    assert expected_record is not None
    driver_record_no_ts = {k: v for k, v in driver_record.items() if k != "timestamp"}
    assert driver_record_no_ts == expected_record, (
        "driver sidecar record must equal extract_probes output"
    )
    assert isinstance(driver_record.get("timestamp"), float)

    # 4. Sanity on the probe shape itself.
    assert driver_record["response_token_ids"] == [19, 198]
    assert driver_record["response_logprobs"] == [-0.1, -0.2]
    assert driver_record["prompt_token_ids"] == [101, 102, 103]
    routed = driver_record["response_routed_experts"]
    assert len(routed) == 2  # trimmed to completion_tokens=2 from a 2-entry buffer
    assert len(routed[0]) == 48
    assert len(routed[0][0]) == 8

    # 5. The driver's parsed JSON response is what the caller's tool
    #    loop consumes — verify it's intact.
    assert rsp["choices"][0]["message"]["content"] == "4"


def test_driver_skips_sidecar_write_when_response_has_no_logprobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the upstream doesn't return logprobs (e.g. the operator forgot
    to patch vLLM), the driver should still return the response so the
    tool loop survives — it just won't write a sidecar line for that
    turn.
    """
    import remote_capture_driver as driver

    sidecar = tmp_path / "probes.jsonl"

    class _FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return json.loads(self.content.decode("utf-8"))

    minimal = json.dumps(
        {
            "id": "x",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    ).encode("utf-8")
    monkeypatch.setattr(
        driver.requests,
        "post",
        lambda url, *, data, headers, timeout: _FakeResponse(minimal),
    )

    rsp = driver._chat_once_with_capture(
        base_url="http://localhost:8000",
        model="Qwen/Qwen3-30B-A3B",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
        timeout=10.0,
        sidecar_path=sidecar,
        prompt_index=0,
    )
    assert rsp["choices"][0]["message"]["content"] == "hi"
    assert not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() == ""
