"""CLI: probe one OpenAI-compatible endpoint and write a contract report.

Usage::

    python -m rollout_market.cli.endpoint_probe \\
        --provider groq \\
        --base-url https://api.groq.com/openai/v1 \\
        --model llama-3.1-70b-versatile \\
        --prompt "Say hi." \\
        --api-key-env GROQ_API_KEY

Writes ``runs/<UTC-timestamp>/endpoint_contract_report.json``. The API key is
read from the named environment variable and never echoed.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..observatory.endpoint_probe import probe_endpoint, write_report


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="endpoint-probe",
        description="Probe a chat-completions endpoint for token/logprob contract coverage.",
    )
    parser.add_argument("--provider", required=True, help="Provider label, e.g. groq, nvidia")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="Model id to send in the request")
    parser.add_argument("--prompt", required=True, help="Single-turn user prompt to send")
    parser.add_argument(
        "--api-key-env",
        required=True,
        help="Environment variable holding the API key (never read inline)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out-root",
        default="runs",
        help="Root directory for report output (default: runs)",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"error: env var {args.api_key_env} is not set; refusing to probe.",
            file=sys.stderr,
        )
        return 2

    sampling: dict[str, object] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
    }
    if args.seed is not None:
        sampling["seed"] = args.seed

    report = probe_endpoint(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        prompt=args.prompt,
        sampling=sampling,
        timeout_s=args.timeout_s,
    )
    out_dir = Path(args.out_root) / _utc_timestamp()
    target = write_report(report, out_dir)
    print(str(target))
    if report.error:
        print(f"warning: probe recorded error: {report.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
