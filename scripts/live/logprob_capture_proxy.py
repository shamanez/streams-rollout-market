"""OpenAI chat-completions proxy that captures rollout-time probes.

Cycle-3 v2 (probe-at-rollout) needs every assistant turn that
hermes-agent emits to carry the rollout-side ``response_logprobs``
(and ideally ``response_token_ids``) that vLLM observed at generation
time.

Patching hermes-agent's batch_runner directly is a maintenance burden;
this proxy is the lowest-friction alternative. It listens on a local
HTTP port, transparently forwards each ``/v1/chat/completions`` POST
to an upstream vLLM endpoint (the SSH-tunnelled spot serve), and:

  * **rewrites the request body** to add ``logprobs: true`` and
    ``top_logprobs: 1`` (so the upstream response includes per-token
    logprob data even when the agent didn't ask for it);
  * **optionally adds** ``extra_body.return_tokens_as_token_ids: true``
    so vLLM returns the integer token IDs alongside each token; this
    is a vLLM-only extension and the upstream ignores it when not
    supported;
  * **appends one JSONL line** to a sidecar file per intercepted
    request, with the per-turn probes plus enough provenance keys
    (``prompt_index``, ``turn_idx``) for downstream pairing.

``prompt_index`` and ``turn_idx`` come from request headers
``X-Prompt-Index`` and ``X-Turn-Idx`` when set (a patched batch_runner
or thin wrapper can stamp them). Both auto-derive in their absence:

  * ``turn_idx`` = count of prior ``role=assistant`` messages in the
    request body (that count equals the index of the turn being
    generated);
  * ``prompt_index`` = hex sha256[:16] of the first ``role=user``
    message in the request body (stable across turns of one task,
    distinct between tasks).

With those defaults the proxy is fully header-free: hermes-agent
runs against it unchanged and the merger pairs probes to ShareGPT
records by recomputing the same hash from each record's first
``from=human`` value.

The pure functions ``inject_logprobs(request_body)`` and
``extract_probes(request_body, response_body)`` carry all the
load-bearing logic and are exhaustively unit-tested; the HTTP wrapper
is a thin shell that wires them together.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def inject_logprobs(request_body: bytes) -> bytes:
    """Return ``request_body`` with ``logprobs=True, top_logprobs=1`` injected.

    Idempotent: if the body already has those keys set, they are
    preserved (the agent's preference wins). The ``extra_body`` dict
    is created if missing and stamped with
    ``return_tokens_as_token_ids: true`` so vLLM returns integer token
    IDs alongside each logprob — non-vLLM upstreams safely ignore the
    unknown key.

    Non-JSON bodies pass through unchanged so the proxy doesn't
    accidentally corrupt streaming uploads or non-chat requests
    (callers should only invoke this on ``/v1/chat/completions``).
    """
    if not request_body:
        return request_body
    try:
        body: dict[str, Any] = json.loads(request_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return request_body
    if not isinstance(body, dict):
        return request_body
    body.setdefault("logprobs", True)
    body.setdefault("top_logprobs", 1)
    extra = body.setdefault("extra_body", {})
    if isinstance(extra, dict):
        extra.setdefault("return_tokens_as_token_ids", True)
    # routed_experts capture is engine-level (the vLLM serve must be
    # launched with `--enable-return-routed-experts`). No request-side
    # parameter is needed; vLLM stamps the routed_experts field on each
    # choice automatically. See
    # https://docs.vllm.ai/en/latest/training/routed_experts_replay/
    return json.dumps(body).encode("utf-8")


def extract_probes(
    request_body: bytes,
    response_body: bytes,
    *,
    prompt_index: int | str,
    turn_idx: int,
) -> dict | None:
    """Build a probe sidecar record from a chat-completions response.

    Reads the OpenAI-shape ``choices[0].logprobs.content`` array (one
    entry per output token, each with ``token``, ``logprob``, and —
    when vLLM is the upstream — a ``token_id`` field surfaced by the
    ``return_tokens_as_token_ids`` extra). Returns ``None`` if the
    response carries no logprobs (e.g. an upstream error or a
    non-chat-completions response slipped through).

    Output schema mirrors the contract documented in
    ``scripts/live/merge_probes_into_trajectories.py``::

        {
          "prompt_index": <int>,
          "turn_idx": <int>,
          "response_token_ids": [<int>, ...] | None,
          "response_logprobs": [<float>, ...],
        }

    ``prompt_token_ids`` is *not* captured here — the chat-completions
    response shape omits the prompt's token ids by default. Capturing
    them requires either ``echo=True`` (legacy ``/v1/completions``) or
    a tokenizer-side re-encoding step; both are deferred to a future
    iteration. The merger script tolerates ``prompt_token_ids=None``
    (downstream callers fall back to chat-template re-encoding).
    """
    try:
        rsp = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(rsp, dict):
        return None
    choices = rsp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    logprobs = choices[0].get("logprobs") if isinstance(choices[0], dict) else None
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    response_logprobs: list[float] = []
    response_token_ids: list[int] = []
    saw_any_token_id = False
    for entry in content:
        if not isinstance(entry, dict):
            return None
        lp = entry.get("logprob")
        if not isinstance(lp, (int, float)):
            return None
        response_logprobs.append(float(lp))
        tid = entry.get("token_id")
        if isinstance(tid, int):
            response_token_ids.append(tid)
            saw_any_token_id = True
    out: dict = {
        "prompt_index": prompt_index,
        "turn_idx": turn_idx,
        "response_logprobs": response_logprobs,
    }
    if saw_any_token_id and len(response_token_ids) == len(response_logprobs):
        out["response_token_ids"] = response_token_ids
    # MoE routed_experts: per the vLLM docs
    # (https://docs.vllm.ai/en/latest/training/routed_experts_replay/),
    # this is a TOP-LEVEL field on each choice, NOT a per-token field
    # under logprobs.content[*]. Shape: ``[gen_len, num_moe_layers,
    # top_k]`` of int16 expert IDs in ``[0, num_experts)``. JSON path:
    # ``choices[0].routed_experts``. vLLM also stamps
    # ``prompt_routed_experts`` at the response top level (shape
    # ``[prompt_len, num_moe_layers, top_k]``) — captured here too so
    # downstream router-pair scripts can build a full RouterTrace
    # against the prompt+response concatenation if they want.
    choice = choices[0] if isinstance(choices[0], dict) else None
    if choice is not None:
        routed = choice.get("routed_experts")
        if isinstance(routed, list) and routed:
            out["response_routed_experts"] = routed
    prompt_routed = rsp.get("prompt_routed_experts")
    if isinstance(prompt_routed, list) and prompt_routed:
        out["prompt_routed_experts"] = prompt_routed
    return out


def derive_turn_idx(request_body: bytes) -> int:
    """Derive the assistant ``turn_idx`` for a chat-completions request.

    The OpenAI chat-completions protocol does not expose request-level
    metadata about *which* turn of an agent loop we're on. But the
    information is implicit in the ``messages`` array: the number of
    prior assistant messages in the request equals the index of the
    assistant turn we're about to generate. The patched batch_runner
    or proxy caller can therefore omit ``X-Turn-Idx`` and rely on
    this derivation.

    Returns ``0`` for non-JSON / malformed bodies (the same fallback
    the legacy header-absent branch produced). Pure function.
    """
    if not request_body:
        return 0
    try:
        body = json.loads(request_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 0
    if not isinstance(body, dict):
        return 0
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")


def derive_prompt_index(request_body: bytes) -> str:
    """Derive a stable per-task ``prompt_index`` from the request body.

    Multi-task hermes-agent runs would otherwise collide on
    ``prompt_index=0`` (the legacy header-absent default), so the
    merger could only pair the first task's probes. Hashing the first
    user message gives a stable string identifier — every turn of the
    same task carries the same first-user message, so all of that
    task's probes group under one ``prompt_index``. Different tasks
    naturally hash to different identifiers without operator wiring.

    Returns ``"0"`` for non-JSON / malformed bodies (string form to
    keep the field type stable across both paths). Pure function.

    The hash is a hex-encoded sha256 truncated to 16 chars — collision
    probability is ~2^-32 across the per-run task set, which is more
    than enough for our 12-task suite.
    """
    if not request_body:
        return "0"
    try:
        body = json.loads(request_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "0"
    if not isinstance(body, dict):
        return "0"
    messages = body.get("messages")
    if not isinstance(messages, list):
        return "0"
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str) and content:
                return _hash_user_message(content)
    return "0"


def _hash_user_message(text: str) -> str:
    """Stable hex-encoded sha256[:16] of a string. Used by both the
    proxy and the merger so the two sides agree on the prompt_index
    derived from a task's first user message."""
    from hashlib import sha256

    return sha256(text.encode("utf-8")).hexdigest()[:16]


def _sidecar_lock_factory() -> threading.Lock:
    return threading.Lock()


class _ProxyHandler(BaseHTTPRequestHandler):
    """Request handler that forwards POST /v1/chat/completions with probes captured.

    Other paths/verbs pass through unchanged. The ``upstream``,
    ``sidecar_path``, ``sidecar_lock`` and ``turn_counter`` are
    attached as class attributes by ``serve()`` before the server
    starts handling requests.
    """

    # Defaults are overridden by serve() below.
    upstream: str = "http://localhost:8000"
    sidecar_path: Path = Path("/tmp/hermes_probes.jsonl")
    sidecar_lock: threading.Lock = _sidecar_lock_factory()
    turn_counter: list[int] = [0]  # mutable container so we can ++ it

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002,D401
        """Silence the default per-request stderr log; we have our own."""

    def _proxy(self, method: str) -> None:
        upstream_url = self.upstream.rstrip("/") + self.path
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b""
        is_chat = method == "POST" and self.path == _CHAT_COMPLETIONS_PATH
        if is_chat:
            body = inject_logprobs(body)

        # Build outbound request with the upstream's expected headers.
        hop_by_hop = {
            "host",
            "content-length",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }
        outbound_headers = {k: v for k, v in self.headers.items() if k.lower() not in hop_by_hop}
        if body:
            outbound_headers["Content-Length"] = str(len(body))
        req = urllib_request.Request(
            upstream_url, data=body if body else None, method=method, headers=outbound_headers
        )
        try:
            with urllib_request.urlopen(req) as resp:
                status = resp.status
                resp_headers = dict(resp.headers.items())
                resp_body = resp.read()
        except urllib_error.HTTPError as exc:
            status = exc.code
            resp_headers = dict(exc.headers.items()) if exc.headers else {}
            resp_body = exc.read() if exc.fp is not None else b""

        # Sidecar capture before we forward — gives the operator the
        # data even if the response write to the client fails.
        if is_chat:
            prompt_hdr = self.headers.get("X-Prompt-Index")
            if prompt_hdr is not None:
                prompt_idx: int | str = int(prompt_hdr) if prompt_hdr.isdigit() else prompt_hdr
            else:
                # Auto-derive a stable per-task identifier so multi-task
                # runs don't all collide on prompt_index=0. The merger
                # uses the same hash of the first user message to pair.
                prompt_idx = derive_prompt_index(body)
            turn_hdr = self.headers.get("X-Turn-Idx")
            if turn_hdr is not None:
                turn_idx = int(turn_hdr)
            else:
                # Auto-derive from the messages array — counting prior
                # assistant messages gives the correct per-task turn
                # index without requiring the caller to wire headers.
                # Note: this assumes one (prompt_index, turn_idx) per
                # request; if the caller batches multiple turns into
                # one request, the X-Turn-Idx header should be set.
                turn_idx = derive_turn_idx(body)
            record = extract_probes(body, resp_body, prompt_index=prompt_idx, turn_idx=turn_idx)
            if record is not None:
                with self.sidecar_lock:
                    with self.sidecar_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")

        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() in hop_by_hop:
                continue
            self.send_header(k, v)
        self.end_headers()
        if resp_body:
            self.wfile.write(resp_body)

    def do_POST(self) -> None:  # noqa: N802
        self._proxy("POST")

    def do_GET(self) -> None:  # noqa: N802
        self._proxy("GET")


def serve(
    *,
    upstream: str,
    sidecar_path: Path,
    host: str = "127.0.0.1",
    port: int = 8001,
) -> None:
    """Run the proxy until KeyboardInterrupt.

    The sidecar file is opened append-only and locked across threads
    so concurrent requests cannot interleave half-written lines.
    """
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.touch(exist_ok=True)
    handler_cls = type(
        "BoundProxyHandler",
        (_ProxyHandler,),
        {
            "upstream": upstream,
            "sidecar_path": sidecar_path,
            "sidecar_lock": _sidecar_lock_factory(),
            "turn_counter": [0],
        },
    )
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    print(
        f"[logprob-proxy] listening on http://{host}:{port}/, "
        f"upstream={upstream}, sidecar={sidecar_path}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[logprob-proxy] shutdown requested", flush=True)
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Local OpenAI chat-completions proxy that injects logprobs=True "
            "and records per-turn probes to a JSONL sidecar."
        )
    )
    parser.add_argument(
        "--upstream",
        default="http://localhost:8000",
        help="Upstream vLLM endpoint (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--sidecar",
        default="/tmp/hermes_probes.jsonl",
        help="Path to the append-only probe sidecar JSONL.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", default=8001, type=int, help="Bind port (default: 8001).")
    args = parser.parse_args(argv)
    serve(
        upstream=args.upstream,
        sidecar_path=Path(args.sidecar),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
