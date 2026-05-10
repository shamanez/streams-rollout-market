"""Endpoint identity gap probe.

Calls an OpenAI-compatible chat-completions endpoint once and records what
the response actually exposes: token IDs, sampled-token logprobs,
top_logprobs, seed acknowledgement, tokenizer hint, raw response hash.

The probe is intentionally read-only and produces no training claims. Its
purpose is to surface the gap between "an endpoint replies" and "an endpoint
returns a behavior policy a trainer can use."
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from pydantic import BaseModel, Field

Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


class EndpointContractReport(BaseModel):
    """Single-call contract coverage record for one provider/model pair."""

    schema_version: Literal["endpoint_probe.v0"] = "endpoint_probe.v0"
    provider: str
    model_label: str
    base_url: str
    sampling_config: dict[str, Any]
    sampling_config_hash: str
    request_hash: str
    response_status: int
    response_byte_size: int
    raw_response_hash: str

    token_ids_available: bool
    sampled_logprobs_available: bool
    top_logprobs_available: bool
    seed_supported: bool

    tokenizer_id: str | None = None
    system_fingerprint: str | None = None
    error: str | None = None

    ran_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _bytes_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def inspect_chat_completion_response(body: Mapping[str, Any]) -> dict[str, Any]:
    """Pure inspector: derive capability flags from a parsed response body.

    Tolerates partial / non-OpenAI-compatible payloads — every flag defaults
    to False when the relevant field is missing rather than raising.
    """
    choices = body.get("choices") or []
    first = choices[0] if choices else {}
    logprobs = first.get("logprobs") if isinstance(first, dict) else None
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    first_token = content[0] if isinstance(content, list) and content else {}

    sampled_logprobs_available = (
        isinstance(content, list) and bool(content) and "logprob" in first_token
    )
    top_logprobs_available = bool(first_token.get("top_logprobs"))
    token_ids_available = bool(first_token.get("token_id") is not None) or (
        "token_ids" in first or "prompt_token_ids" in body
    )
    system_fingerprint = body.get("system_fingerprint")
    tokenizer_id = body.get("tokenizer") or body.get("model")
    return {
        "token_ids_available": bool(token_ids_available),
        "sampled_logprobs_available": bool(sampled_logprobs_available),
        "top_logprobs_available": bool(top_logprobs_available),
        "system_fingerprint": system_fingerprint if isinstance(system_fingerprint, str) else None,
        "tokenizer_id": tokenizer_id if isinstance(tokenizer_id, str) else None,
    }


def _urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout_s: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() if exc.fp is not None else b""


def probe_endpoint(
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    sampling: Mapping[str, Any] | None = None,
    *,
    timeout_s: float = 30.0,
    transport: Transport | None = None,
) -> EndpointContractReport:
    """Probe one OpenAI-compatible endpoint and return a contract coverage report.

    `transport` is a seam for tests; production calls use urllib over HTTPS.
    The api_key is sent only in the Authorization header and never recorded
    in the report or in any hash.
    """
    sampling_config: dict[str, Any] = dict(sampling or {})
    sampling_config.setdefault("temperature", 0.7)
    sampling_config.setdefault("max_tokens", 32)
    sampling_config.setdefault("logprobs", True)
    sampling_config.setdefault("top_logprobs", 5)
    seed_requested = "seed" in sampling_config

    request_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **sampling_config,
    }
    body = json.dumps(request_payload).encode("utf-8")
    # Many providers (Cerebras, Groq) front their API with Cloudflare, which
    # rejects the default `Python-urllib/...` User-Agent with error 1010.
    # We send a generic UA the probe owns so requests are not silently blocked.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "rollout-market-endpoint-probe/0.1",
        "Accept": "application/json",
    }
    url = base_url.rstrip("/") + "/chat/completions"
    request_hash = _stable_hash(
        {"provider": provider, "url": url, "payload": request_payload}
    )
    sampling_hash = _stable_hash(sampling_config)

    transport = transport or _urllib_transport
    status: int = 0
    raw_bytes = b""
    error: str | None = None
    try:
        status, raw_bytes = transport(url, headers, body, timeout_s)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    parsed: dict[str, Any] = {}
    if raw_bytes:
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = error or f"response not JSON: {type(exc).__name__}"
            parsed = {}

    flags = inspect_chat_completion_response(parsed)
    seed_supported = seed_requested and flags["system_fingerprint"] is not None

    return EndpointContractReport(
        provider=provider,
        model_label=model,
        base_url=base_url,
        sampling_config=sampling_config,
        sampling_config_hash=sampling_hash,
        request_hash=request_hash,
        response_status=status,
        response_byte_size=len(raw_bytes),
        raw_response_hash=_bytes_hash(raw_bytes),
        token_ids_available=flags["token_ids_available"],
        sampled_logprobs_available=flags["sampled_logprobs_available"],
        top_logprobs_available=flags["top_logprobs_available"],
        seed_supported=seed_supported,
        tokenizer_id=flags["tokenizer_id"],
        system_fingerprint=flags["system_fingerprint"],
        error=error,
    )


def write_report(report: EndpointContractReport, out_dir: Path | str) -> Path:
    """Write the report as endpoint_contract_report.json under out_dir.

    out_dir is created if missing. Returns the full path written.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "endpoint_contract_report.json"
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return target
