"""Tests for scripts/live/logprob_capture_proxy.py.

The pure transform functions carry all the load-bearing logic; they
get exhaustive unit tests. The HTTP wrapper gets one end-to-end
integration test against a stdlib-only fake upstream so we don't
need to mock urllib.

Tests cover:

  * inject_logprobs: idempotent merge, preserves agent's preference,
    survives non-JSON bodies, stamps the vLLM extra_body key.
  * extract_probes: vLLM-shape response with token_ids, OpenAI-shape
    without token_ids, empty/missing content branches, non-JSON
    responses return None.
  * end-to-end: a fake upstream replies with a logprobs payload; the
    proxy forwards the rewritten request, records a sidecar line, and
    returns the upstream's response body verbatim to the client.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType
from urllib import request as urllib_request

import pytest


def _load_module() -> ModuleType:
    spec_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "live" / "logprob_capture_proxy.py"
    )
    spec = importlib.util.spec_from_file_location("logprob_capture_proxy", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def proxy() -> ModuleType:
    return _load_module()


# --- inject_logprobs --------------------------------------------------------


def test_inject_logprobs_adds_missing_keys(proxy: ModuleType) -> None:
    body = json.dumps({"model": "Qwen/Qwen3-32B", "messages": []}).encode("utf-8")
    out = json.loads(proxy.inject_logprobs(body).decode())
    assert out["logprobs"] is True
    assert out["top_logprobs"] == 1
    assert out["extra_body"]["return_tokens_as_token_ids"] is True


def test_inject_logprobs_preserves_agent_preference(proxy: ModuleType) -> None:
    """If the agent already set logprobs=False, the proxy must not
    override — the agent's choice wins on every load-bearing key."""
    body = json.dumps(
        {
            "model": "Qwen/Qwen3-32B",
            "logprobs": False,
            "top_logprobs": 5,
            "extra_body": {"return_tokens_as_token_ids": False},
        }
    ).encode("utf-8")
    out = json.loads(proxy.inject_logprobs(body).decode())
    assert out["logprobs"] is False
    assert out["top_logprobs"] == 5
    assert out["extra_body"]["return_tokens_as_token_ids"] is False


def test_inject_logprobs_passes_non_json_through(proxy: ModuleType) -> None:
    raw = b"not json at all"
    assert proxy.inject_logprobs(raw) == raw


def test_inject_logprobs_passes_empty_body_through(proxy: ModuleType) -> None:
    assert proxy.inject_logprobs(b"") == b""


# --- extract_probes ---------------------------------------------------------


def _vllm_response(*, with_token_ids: bool = True) -> bytes:
    content = [
        {
            "token": "The",
            "logprob": -0.1,
            **({"token_id": 791} if with_token_ids else {}),
        },
        {
            "token": " answer",
            "logprob": -0.2,
            **({"token_id": 4320} if with_token_ids else {}),
        },
        {
            "token": " 4",
            "logprob": -0.05,
            **({"token_id": 19} if with_token_ids else {}),
        },
    ]
    rsp = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The answer 4"},
                "logprobs": {"content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return json.dumps(rsp).encode("utf-8")


def test_extract_probes_vllm_shape_with_token_ids(proxy: ModuleType) -> None:
    out = proxy.extract_probes(
        b"{}", _vllm_response(with_token_ids=True), prompt_index=0, turn_idx=0
    )
    assert out is not None
    assert out["prompt_index"] == 0
    assert out["turn_idx"] == 0
    assert out["response_logprobs"] == [-0.1, -0.2, -0.05]
    assert out["response_token_ids"] == [791, 4320, 19]


def test_extract_probes_openai_shape_no_token_ids(proxy: ModuleType) -> None:
    """Vanilla OpenAI chat-completions doesn't carry per-token ids;
    we still return the logprobs and just omit token_ids."""
    out = proxy.extract_probes(
        b"{}", _vllm_response(with_token_ids=False), prompt_index=1, turn_idx=3
    )
    assert out is not None
    assert out["prompt_index"] == 1
    assert out["turn_idx"] == 3
    assert out["response_logprobs"] == [-0.1, -0.2, -0.05]
    assert "response_token_ids" not in out


def test_extract_probes_returns_none_on_no_logprobs(proxy: ModuleType) -> None:
    rsp = json.dumps({"choices": [{"index": 0, "message": {"content": "x"}}]}).encode()
    assert proxy.extract_probes(b"{}", rsp, prompt_index=0, turn_idx=0) is None


def test_extract_probes_returns_none_on_empty_choices(proxy: ModuleType) -> None:
    rsp = json.dumps({"choices": []}).encode()
    assert proxy.extract_probes(b"{}", rsp, prompt_index=0, turn_idx=0) is None


def test_extract_probes_returns_none_on_non_json(proxy: ModuleType) -> None:
    assert proxy.extract_probes(b"{}", b"not json", prompt_index=0, turn_idx=0) is None


def test_extract_probes_returns_none_on_malformed_entry(proxy: ModuleType) -> None:
    """A non-numeric logprob in any entry returns None rather than
    a partially populated record."""
    rsp = json.dumps(
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "a", "logprob": -0.1},
                            {"token": "b", "logprob": "oops"},
                        ]
                    }
                }
            ]
        }
    ).encode()
    assert proxy.extract_probes(b"{}", rsp, prompt_index=0, turn_idx=0) is None


# --- end-to-end -------------------------------------------------------------


class _FakeUpstream:
    """A tiny stdlib HTTP server that echoes a vLLM-style chat-completions
    response and records the rewritten request body for inspection."""

    def __init__(self) -> None:
        self.received_body: bytes = b""

    def start(self) -> tuple[HTTPServer, int]:
        rec = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *a, **kw) -> None:  # noqa
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                rec.received_body = self.rfile.read(n)
                resp = _vllm_response(with_token_ids=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, port


def test_end_to_end_proxy_records_sidecar(proxy: ModuleType, tmp_path: Path) -> None:
    upstream = _FakeUpstream()
    server, up_port = upstream.start()
    sidecar = tmp_path / "probes.jsonl"

    # Boot the proxy on a free port in a background thread.
    handler_cls = type(
        "ProxyHandlerForTest",
        (proxy._ProxyHandler,),
        {
            "upstream": f"http://127.0.0.1:{up_port}",
            "sidecar_path": sidecar,
            "sidecar_lock": proxy._sidecar_lock_factory(),
            "turn_counter": [0],
        },
    )
    proxy_server = proxy.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    proxy_port = proxy_server.server_address[1]
    t = threading.Thread(target=proxy_server.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib_request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "Qwen/Qwen3-32B",
                    "messages": [{"role": "user", "content": "Q"}],
                }
            ).encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Prompt-Index": "7",
                "X-Turn-Idx": "2",
            },
        )
        with urllib_request.urlopen(req) as resp:
            body = resp.read()
    finally:
        proxy_server.shutdown()
        proxy_server.server_close()
        server.shutdown()
        server.server_close()

    # Upstream saw the rewritten body with logprobs injected.
    received = json.loads(upstream.received_body.decode())
    assert received["logprobs"] is True
    assert received["top_logprobs"] == 1
    assert received["extra_body"]["return_tokens_as_token_ids"] is True

    # Client got the upstream's response verbatim.
    assert json.loads(body)["choices"][0]["finish_reason"] == "stop"

    # Sidecar carries one line with the captured probes.
    lines = [ln for ln in sidecar.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["prompt_index"] == 7
    assert rec["turn_idx"] == 2
    assert rec["response_logprobs"] == [-0.1, -0.2, -0.05]
    assert rec["response_token_ids"] == [791, 4320, 19]
