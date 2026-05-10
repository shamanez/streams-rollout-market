import json
from pathlib import Path
from urllib.error import URLError

import pytest

from rollout_market.cli import endpoint_probe as cli
from rollout_market.observatory.endpoint_probe import (
    EndpointContractReport,
    inspect_chat_completion_response,
    probe_endpoint,
    write_report,
)


def _full_response() -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model-1",
        "system_fingerprint": "fp_demo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "logprobs": {
                    "content": [
                        {
                            "token": "ok",
                            "logprob": -0.123,
                            "top_logprobs": [
                                {"token": "ok", "logprob": -0.123},
                                {"token": "yes", "logprob": -2.4},
                            ],
                        }
                    ]
                },
                "finish_reason": "stop",
            }
        ],
    }


def _minimal_response() -> dict:
    return {
        "id": "chatcmpl-2",
        "model": "test-model-2",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
    }


def _make_transport(status: int, body: dict | bytes):
    payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    captured: dict = {}

    def transport(url, headers, request_body, timeout_s):
        captured["url"] = url
        captured["headers"] = dict(headers)
        captured["body"] = request_body
        captured["timeout"] = timeout_s
        return status, payload

    transport.captured = captured  # type: ignore[attr-defined]
    return transport


def test_inspect_full_response_flags_all_capabilities():
    flags = inspect_chat_completion_response(_full_response())
    assert flags["sampled_logprobs_available"] is True
    assert flags["top_logprobs_available"] is True
    assert flags["system_fingerprint"] == "fp_demo"
    assert flags["tokenizer_id"] == "test-model-1"
    assert flags["token_ids_available"] is False


def test_inspect_minimal_response_flags_no_logprobs():
    flags = inspect_chat_completion_response(_minimal_response())
    assert flags["sampled_logprobs_available"] is False
    assert flags["top_logprobs_available"] is False
    assert flags["system_fingerprint"] is None
    assert flags["token_ids_available"] is False


def test_inspect_handles_token_id_field():
    body = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "ok",
                            "token_id": 12345,
                            "logprob": -0.5,
                            "top_logprobs": [],
                        }
                    ]
                }
            }
        ]
    }
    flags = inspect_chat_completion_response(body)
    assert flags["token_ids_available"] is True
    assert flags["sampled_logprobs_available"] is True
    assert flags["top_logprobs_available"] is False


def test_probe_endpoint_records_full_capabilities_with_seed():
    transport = _make_transport(200, _full_response())
    report = probe_endpoint(
        provider="fake",
        base_url="https://api.example.com/v1",
        model="test-model-1",
        api_key="not-a-real-key",
        prompt="hi",
        sampling={"seed": 1234, "temperature": 0.7, "max_tokens": 8},
        transport=transport,
    )
    assert isinstance(report, EndpointContractReport)
    assert report.response_status == 200
    assert report.sampled_logprobs_available is True
    assert report.top_logprobs_available is True
    assert report.seed_supported is True
    assert report.system_fingerprint == "fp_demo"
    assert report.raw_response_hash.startswith("sha256:")
    assert report.request_hash.startswith("sha256:")
    assert "Authorization" in transport.captured["headers"]
    assert transport.captured["headers"]["Authorization"] == "Bearer not-a-real-key"
    assert transport.captured["url"].endswith("/chat/completions")
    # Cerebras / Groq sit behind Cloudflare, which 1010s the default
    # `Python-urllib/...` User-Agent. The probe must send its own UA.
    assert "User-Agent" in transport.captured["headers"]
    assert "rollout-market" in transport.captured["headers"]["User-Agent"]


def test_probe_endpoint_seed_unsupported_when_no_fingerprint():
    transport = _make_transport(200, _minimal_response())
    report = probe_endpoint(
        provider="fake",
        base_url="https://api.example.com/v1",
        model="test-model-2",
        api_key="x",
        prompt="hi",
        sampling={"seed": 1, "max_tokens": 8},
        transport=transport,
    )
    assert report.seed_supported is False
    assert report.sampled_logprobs_available is False
    assert report.top_logprobs_available is False


def test_probe_endpoint_records_transport_error():
    def transport(url, headers, body, timeout_s):
        raise URLError("connection refused")

    report = probe_endpoint(
        provider="fake",
        base_url="https://api.example.com/v1",
        model="m",
        api_key="x",
        prompt="hi",
        transport=transport,
    )
    assert report.response_status == 0
    assert report.error is not None
    assert "URLError" in report.error
    assert report.sampled_logprobs_available is False


def test_probe_endpoint_records_non_json_body():
    transport = _make_transport(503, b"<html>oops</html>")
    report = probe_endpoint(
        provider="fake",
        base_url="https://api.example.com/v1",
        model="m",
        api_key="x",
        prompt="hi",
        transport=transport,
    )
    assert report.response_status == 503
    assert report.error is not None
    assert report.raw_response_hash.startswith("sha256:")


def test_request_hash_excludes_api_key():
    t1 = _make_transport(200, _minimal_response())
    t2 = _make_transport(200, _minimal_response())
    r1 = probe_endpoint("p", "https://x", "m", "key-1", "hi", transport=t1)
    r2 = probe_endpoint("p", "https://x", "m", "key-2", "hi", transport=t2)
    assert r1.request_hash == r2.request_hash


def test_write_report_creates_directory(tmp_path: Path):
    transport = _make_transport(200, _minimal_response())
    report = probe_endpoint("p", "https://x", "m", "k", "hi", transport=transport)
    out_dir = tmp_path / "runs" / "20260510T000000Z"
    target = write_report(report, out_dir)
    assert target == out_dir / "endpoint_contract_report.json"
    payload = json.loads(target.read_text())
    assert payload["schema_version"] == "endpoint_probe.v0"
    assert payload["provider"] == "p"


def test_cli_main_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    transport = _make_transport(200, _full_response())
    captured: dict = {}

    def fake_probe(**kwargs):
        captured.update(kwargs)
        return probe_endpoint(transport=transport, **kwargs)

    monkeypatch.setattr(cli, "probe_endpoint", fake_probe)
    monkeypatch.setenv("FAKE_KEY", "secret-do-not-log")

    rc = cli.main(
        [
            "--provider",
            "fake",
            "--base-url",
            "https://api.example.com/v1",
            "--model",
            "test-model-1",
            "--prompt",
            "hi",
            "--api-key-env",
            "FAKE_KEY",
            "--out-root",
            str(tmp_path / "runs"),
            "--seed",
            "7",
        ]
    )
    assert rc == 0
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    target = runs[0] / "endpoint_contract_report.json"
    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["provider"] == "fake"
    assert payload["model_label"] == "test-model-1"
    assert payload["seed_supported"] is True
    assert "secret-do-not-log" not in target.read_text()


def test_cli_main_refuses_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    rc = cli.main(
        [
            "--provider",
            "p",
            "--base-url",
            "https://x",
            "--model",
            "m",
            "--prompt",
            "hi",
            "--api-key-env",
            "MISSING_KEY",
            "--out-root",
            str(tmp_path / "runs"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "MISSING_KEY" in err


def test_acceptance_report_contains_required_fields():
    transport = _make_transport(200, _full_response())
    report = probe_endpoint(
        provider="groq-like",
        base_url="https://api.example.com/v1",
        model="some-model",
        api_key="x",
        prompt="hi",
        sampling={"seed": 42},
        transport=transport,
    )
    payload = report.model_dump(mode="json")
    for field in (
        "provider",
        "model_label",
        "token_ids_available",
        "sampled_logprobs_available",
        "top_logprobs_available",
        "seed_supported",
        "raw_response_hash",
    ):
        assert field in payload, f"missing required field: {field}"
