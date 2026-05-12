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
    # Top-level return_token_ids surfaces prompt_token_ids /
    # choices[i].token_ids as raw int lists — see vLLM
    # entrypoints/openai/chat_completion/protocol.py.
    assert out["return_token_ids"] is True


def test_inject_logprobs_preserves_agent_preference(proxy: ModuleType) -> None:
    """If the agent already set logprobs=False, the proxy must not
    override — the agent's choice wins on every load-bearing key."""
    body = json.dumps(
        {
            "model": "Qwen/Qwen3-32B",
            "logprobs": False,
            "top_logprobs": 5,
            "return_token_ids": False,
        }
    ).encode("utf-8")
    out = json.loads(proxy.inject_logprobs(body).decode())
    assert out["logprobs"] is False
    assert out["top_logprobs"] == 5
    assert out["return_token_ids"] is False


def test_inject_logprobs_passes_non_json_through(proxy: ModuleType) -> None:
    raw = b"not json at all"
    assert proxy.inject_logprobs(raw) == raw


def test_inject_logprobs_passes_empty_body_through(proxy: ModuleType) -> None:
    assert proxy.inject_logprobs(b"") == b""


def test_inject_logprobs_does_not_add_routed_experts_request_param(
    proxy: ModuleType,
) -> None:
    """Per the vLLM docs, routed_experts capture is an engine-level
    flag (``--enable-return-routed-experts``), NOT a per-request
    parameter. The proxy must not invent an ``extra_body.return_routed_experts``
    key — earlier versions did and that was a documentation-time bug."""
    body = json.dumps({"model": "Qwen/Qwen3-30B-A3B", "messages": []}).encode("utf-8")
    out = json.loads(proxy.inject_logprobs(body).decode())
    extra = out.get("extra_body", {})
    assert "return_routed_experts" not in extra
    # return_token_ids is the load-bearing opt-in (top-level, not extra_body).
    assert out["return_token_ids"] is True


def test_inject_disables_thinking_by_default(proxy: ModuleType) -> None:
    """Qwen3 (both Dense and MoE) defaults to reasoning mode; the
    ``<think>...</think>`` blocks blew past 131K context in 3 turns
    during the 2026-05-12 Hermes-MoE summarize-repo smoke. The only
    documented off-switch is ``chat_template_kwargs.enable_thinking=
    False`` at the OpenAI body level — Hermes-Agent doesn't thread it
    through, so the proxy injects it on every outbound request."""
    body = json.dumps({"model": "Qwen/Qwen3-30B-A3B", "messages": []}).encode("utf-8")
    out = json.loads(proxy.inject_logprobs(body).decode())
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_inject_preserves_caller_thinking_choice(proxy: ModuleType) -> None:
    """Idempotent: if the caller explicitly opts into reasoning mode
    (``enable_thinking=True``) or has other ``chat_template_kwargs``
    keys set, the proxy must NOT overwrite them — same agent-preference
    rule as for ``logprobs`` / ``top_logprobs``."""
    body = json.dumps(
        {
            "model": "Qwen/Qwen3-30B-A3B",
            "messages": [],
            "chat_template_kwargs": {"enable_thinking": True, "custom_key": "value"},
        }
    ).encode("utf-8")
    out = json.loads(proxy.inject_logprobs(body).decode())
    assert out["chat_template_kwargs"]["enable_thinking"] is True
    assert out["chat_template_kwargs"]["custom_key"] == "value"


# --- derive_turn_idx --------------------------------------------------------


def test_derive_turn_idx_counts_prior_assistant_messages(proxy: ModuleType) -> None:
    """The turn index for the next-to-be-generated assistant turn is
    exactly the count of prior assistant messages in the request's
    messages array."""
    body = json.dumps(
        {
            "model": "Qwen/Qwen3-32B",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "first answer"},
                {"role": "tool", "content": "tool result"},
                {"role": "assistant", "content": "second answer"},
                {"role": "tool", "content": "tool result 2"},
            ],
        }
    ).encode()
    # Two prior assistants -> we're about to generate turn 2.
    assert proxy.derive_turn_idx(body) == 2


def test_derive_turn_idx_first_request_yields_zero(proxy: ModuleType) -> None:
    """The very first chat-completions call for a task has no prior
    assistant messages — turn_idx must be 0."""
    body = json.dumps(
        {
            "model": "Qwen/Qwen3-32B",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
        }
    ).encode()
    assert proxy.derive_turn_idx(body) == 0


def test_derive_turn_idx_handles_missing_messages(proxy: ModuleType) -> None:
    """A body without a ``messages`` field (or non-JSON) falls back
    to 0 — never crashes."""
    assert proxy.derive_turn_idx(b'{"model": "x"}') == 0
    assert proxy.derive_turn_idx(b"not json") == 0
    assert proxy.derive_turn_idx(b"") == 0


# --- derive_prompt_index ----------------------------------------------------


def test_derive_prompt_index_hashes_first_user_message(proxy: ModuleType) -> None:
    """The hash of the first user message is the stable per-task
    identifier. Same task across turns → same hash. Different tasks
    → different hashes."""
    body_a = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "what is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
    ).encode()
    body_b = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "what is 2+2?"},
            ]
        }
    ).encode()
    body_other = json.dumps({"messages": [{"role": "user", "content": "what is 3+3?"}]}).encode()
    h_a = proxy.derive_prompt_index(body_a)
    h_b = proxy.derive_prompt_index(body_b)
    h_other = proxy.derive_prompt_index(body_other)
    # Same task, different turn-states → same hash.
    assert h_a == h_b
    # Different task → different hash.
    assert h_a != h_other
    # Hash is a 16-char hex string.
    assert len(h_a) == 16
    assert all(c in "0123456789abcdef" for c in h_a)


def test_derive_prompt_index_fallback_zero_string(proxy: ModuleType) -> None:
    """Bodies without a user message or non-JSON fall back to '0'
    (string form keeps the field type stable)."""
    assert proxy.derive_prompt_index(b"") == "0"
    assert proxy.derive_prompt_index(b"not json") == "0"
    assert proxy.derive_prompt_index(b'{"messages": []}') == "0"
    assert proxy.derive_prompt_index(b'{"messages": [{"role": "system", "content": "x"}]}') == "0"


# --- extract_probes ---------------------------------------------------------


def _vllm_response(*, with_token_ids: bool = True) -> bytes:
    """Build a synthetic vLLM chat-completions response.

    When ``with_token_ids=True`` the response includes the top-level
    ``prompt_token_ids`` and per-choice ``token_ids`` fields that vLLM
    populates when the request had ``return_token_ids=true``.
    """
    content = [
        {"token": "The", "logprob": -0.1},
        {"token": " answer", "logprob": -0.2},
        {"token": " 4", "logprob": -0.05},
    ]
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "The answer 4"},
        "logprobs": {"content": content},
        "finish_reason": "stop",
    }
    rsp = {"id": "chatcmpl-1", "choices": [choice]}
    if with_token_ids:
        choice["token_ids"] = [791, 4320, 19]
        rsp["prompt_token_ids"] = [100, 101, 102, 103]
    return json.dumps(rsp).encode("utf-8")


def test_extract_probes_vllm_shape_with_token_ids(proxy: ModuleType) -> None:
    """When the response includes vLLM's return_token_ids fields,
    both prompt_token_ids (top-level) and response_token_ids
    (choices[0].token_ids) land on the sidecar."""
    out = proxy.extract_probes(
        b"{}", _vllm_response(with_token_ids=True), prompt_index=0, turn_idx=0
    )
    assert out is not None
    assert out["prompt_index"] == 0
    assert out["turn_idx"] == 0
    assert out["response_logprobs"] == [-0.1, -0.2, -0.05]
    assert out["response_token_ids"] == [791, 4320, 19]
    assert out["prompt_token_ids"] == [100, 101, 102, 103]


def test_extract_probes_openai_shape_no_token_ids(proxy: ModuleType) -> None:
    """Vanilla OpenAI chat-completions doesn't carry per-token ids;
    we still return the logprobs and just omit the token_ids keys."""
    out = proxy.extract_probes(
        b"{}", _vllm_response(with_token_ids=False), prompt_index=1, turn_idx=3
    )
    assert out is not None
    assert out["prompt_index"] == 1
    assert out["turn_idx"] == 3
    assert out["response_logprobs"] == [-0.1, -0.2, -0.05]
    assert "response_token_ids" not in out
    assert "prompt_token_ids" not in out


def test_extract_probes_captures_routed_experts_when_present(
    proxy: ModuleType,
) -> None:
    """vLLM stamps ``routed_experts`` as a TOP-LEVEL field on each
    choice (NOT under logprobs.content[*]) — shape
    ``[gen_len, num_moe_layers, top_k]``. See
    https://docs.vllm.ai/en/latest/training/routed_experts_replay/.
    """
    rsp = json.dumps(
        {
            "prompt_token_ids": [50, 51, 52],
            "prompt_routed_experts": [
                [[7, 6], [5, 4], [3, 2]],
                [[6, 7], [4, 5], [2, 3]],
            ],
            "choices": [
                {
                    "token_ids": [100, 101],
                    "logprobs": {
                        "content": [
                            {"token": "A", "logprob": -0.1},
                            {"token": "B", "logprob": -0.2},
                        ]
                    },
                    "routed_experts": [
                        [[0, 1], [2, 3], [4, 5]],
                        [[1, 0], [3, 2], [5, 4]],
                    ],
                }
            ],
        }
    ).encode()
    out = proxy.extract_probes(b"{}", rsp, prompt_index=0, turn_idx=0)
    assert out is not None
    assert out["response_routed_experts"] == [
        [[0, 1], [2, 3], [4, 5]],
        [[1, 0], [3, 2], [5, 4]],
    ]
    assert out["prompt_routed_experts"] == [
        [[7, 6], [5, 4], [3, 2]],
        [[6, 7], [4, 5], [2, 3]],
    ]
    assert out["response_token_ids"] == [100, 101]
    assert out["prompt_token_ids"] == [50, 51, 52]


def test_extract_probes_omits_routed_experts_when_absent(
    proxy: ModuleType,
) -> None:
    """Dense responses (no routed_experts) should produce a sidecar
    record WITHOUT the response_routed_experts key — the AgentStep
    validator forbids degenerate empty lists on a tool step and the
    downstream merger handles None."""
    rsp = json.dumps(
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "A", "logprob": -0.1, "token_id": 100},
                        ]
                    }
                }
            ]
        }
    ).encode()
    out = proxy.extract_probes(b"{}", rsp, prompt_index=0, turn_idx=0)
    assert out is not None
    assert "response_routed_experts" not in out


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

    # Upstream saw the rewritten body with logprobs + return_token_ids injected.
    received = json.loads(upstream.received_body.decode())
    assert received["logprobs"] is True
    assert received["top_logprobs"] == 1
    assert received["return_token_ids"] is True

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
    assert rec["prompt_token_ids"] == [100, 101, 102, 103]


def test_end_to_end_proxy_auto_derives_turn_idx_without_header(
    proxy: ModuleType, tmp_path: Path
) -> None:
    """When the caller doesn't set X-Turn-Idx, the proxy must auto-
    derive it by counting prior assistant messages in the request's
    messages array — so multi-turn agent loops record correct turn
    indices without header wiring."""
    upstream = _FakeUpstream()
    server, up_port = upstream.start()
    sidecar = tmp_path / "probes.jsonl"

    handler_cls = type(
        "ProxyHandlerNoHeader",
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
    threading.Thread(target=proxy_server.serve_forever, daemon=True).start()
    try:
        # The request has 2 prior assistant messages → next is turn 2.
        req = urllib_request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "Qwen/Qwen3-32B",
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "turn 0"},
                        {"role": "tool", "content": "r1"},
                        {"role": "assistant", "content": "turn 1"},
                        {"role": "tool", "content": "r2"},
                    ],
                }
            ).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib_request.urlopen(req).read()
    finally:
        proxy_server.shutdown()
        proxy_server.server_close()
        server.shutdown()
        server.server_close()

    lines = [ln for ln in sidecar.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # X-Prompt-Index absent → auto-derived as hex hash of first user msg.
    assert isinstance(rec["prompt_index"], str)
    assert len(rec["prompt_index"]) == 16
    assert all(c in "0123456789abcdef" for c in rec["prompt_index"])
    # turn_idx auto-derived from 2 prior assistant messages.
    assert rec["turn_idx"] == 2
