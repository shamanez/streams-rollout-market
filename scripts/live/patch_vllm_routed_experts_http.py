#!/usr/bin/env python3
"""Patch an installed vLLM wheel to serialize ``CompletionOutput.routed_experts``
over the OpenAI HTTP shim.

The engine in the installed wheel already captures generation-side routing
into ``CompletionOutput.routed_experts`` (a numpy ``[gen_len, num_layers,
top_k]`` int16 array). The wheel's ``entrypoints/openai`` layer simply
forgets to copy that field into the JSON response.

This script adds a single optional ``routed_experts`` field to the two
non-streaming response-choice models (``CompletionResponseChoice`` and
``ChatCompletionResponseChoice``) and threads ``output.routed_experts``
into the three sites where those models are constructed.

Prompt-side ``prompt_routed_experts`` is intentionally skipped: the
installed wheel's ``RequestOutput`` does not carry that field, so we
restrict scope to what the engine actually emits today.

The script is idempotent: it inspects each target file, applies the
edit only if missing, and otherwise reports ``already-patched``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _replace_once(text: str, marker: str, insertion: str, sentinel: str) -> str:
    """Insert ``insertion`` right after ``marker`` in ``text``.

    Asserts exactly one occurrence of ``marker``. If ``sentinel`` is
    already present, returns ``text`` unchanged (idempotent).
    """
    if sentinel in text:
        return text  # already patched
    occurrences = text.count(marker)
    if occurrences != 1:
        raise SystemExit(
            f"patch: expected exactly one occurrence of marker, got {occurrences}\n"
            f"marker={marker!r}"
        )
    return text.replace(marker, marker + insertion, 1)


def _replace_all(text: str, marker: str, insertion: str, sentinel: str, expected: int) -> str:
    if sentinel in text:
        return text
    occurrences = text.count(marker)
    if occurrences != expected:
        raise SystemExit(
            f"patch: expected {expected} occurrences, got {occurrences}\nmarker={marker!r}"
        )
    return text.replace(marker, marker + insertion)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <site-packages/vllm>", file=sys.stderr)
        raise SystemExit(2)
    vllm_root = Path(sys.argv[1])
    if not vllm_root.is_dir():
        raise SystemExit(f"not a directory: {vllm_root}")

    patched: list[str] = []
    skipped: list[str] = []

    # 1. completion/protocol.py — add field to CompletionResponseChoice
    p = vllm_root / "entrypoints/openai/completion/protocol.py"
    src = p.read_text(encoding="utf-8")
    if "routed_experts" in src:
        skipped.append(str(p))
    else:
        marker = "    prompt_token_ids: list[int] | None = None  # For prompt\n"
        addition = (
            "    routed_experts: list[list[list[int]]] | None = (\n"
            "        None  # [gen_len, num_layers, top_k] when --enable-return-routed-experts\n"
            "    )\n"
        )
        new = _replace_once(src, marker, addition, sentinel="routed_experts")
        p.write_text(new, encoding="utf-8")
        patched.append(str(p))

    # 2. completion/serving.py — populate field in CompletionResponseChoice ctor
    p = vllm_root / "entrypoints/openai/completion/serving.py"
    src = p.read_text(encoding="utf-8")
    if "routed_experts=" in src:
        skipped.append(str(p))
    else:
        marker = (
            "                    token_ids=(\n"
            "                        as_list(output.token_ids) if request.return_token_ids else None\n"
            "                    ),\n"
        )
        addition = (
            "                    routed_experts=(\n"
            "                        output.routed_experts.tolist()\n"
            "                        if getattr(output, \"routed_experts\", None) is not None\n"
            "                        else None\n"
            "                    ),\n"
        )
        new = _replace_once(src, marker, addition, sentinel="routed_experts=(")
        p.write_text(new, encoding="utf-8")
        patched.append(str(p))

    # 3. chat_completion/protocol.py — add field to ChatCompletionResponseChoice
    p = vllm_root / "entrypoints/openai/chat_completion/protocol.py"
    src = p.read_text(encoding="utf-8")
    if "routed_experts" in src:
        skipped.append(str(p))
    else:
        # Anchor: the existing ``token_ids: list[int] | None = None`` line in
        # ChatCompletionResponseChoice. There are several token_ids lines in
        # the file — match the unique one with this exact prefix comment.
        marker = (
            "    # not part of the OpenAI spec but is useful for tracing the tokens\n"
            "    # in agent scenarios\n"
            "    token_ids: list[int] | None = None\n"
        )
        addition = (
            "    routed_experts: list[list[list[int]]] | None = (\n"
            "        None  # [gen_len, num_layers, top_k] when --enable-return-routed-experts\n"
            "    )\n"
        )
        new = _replace_once(src, marker, addition, sentinel="routed_experts")
        p.write_text(new, encoding="utf-8")
        patched.append(str(p))

    # 4. chat_completion/serving.py — populate at both non-streaming ctor sites.
    # The two ctor blocks live at different indentation levels (one inside a
    # deeper nested context). Patch each at its own indent.
    p = vllm_root / "entrypoints/openai/chat_completion/serving.py"
    src = p.read_text(encoding="utf-8")
    if "routed_experts=" in src:
        skipped.append(str(p))
    else:

        def _ctor_marker(field_indent: str) -> str:
            return (
                f"{field_indent}stop_reason=output.stop_reason,\n"
                f"{field_indent}token_ids=(\n"
                f"{field_indent}    as_list(output.token_ids) if request.return_token_ids else None\n"
                f"{field_indent}),\n"
            )

        def _ctor_addition(field_indent: str) -> str:
            return (
                f"{field_indent}routed_experts=(\n"
                f"{field_indent}    output.routed_experts.tolist()\n"
                f'{field_indent}    if getattr(output, "routed_experts", None) is not None\n'
                f"{field_indent}    else None\n"
                f"{field_indent}),\n"
            )

        new = src
        # Deeper site (inside ``for output in ...:`` inside a guard) — 20 spaces.
        new = _replace_once(
            new,
            _ctor_marker(" " * 20),
            _ctor_addition(" " * 20),
            sentinel="routed_experts=(",
        )
        # Outer site — 16 spaces.
        new = _replace_once(
            new,
            _ctor_marker(" " * 16),
            _ctor_addition(" " * 16),
            sentinel="placeholder-does-not-exist-yet",  # bypass idempotency check after first patch
        )
        p.write_text(new, encoding="utf-8")
        patched.append(str(p))

    print("patched:")
    for f in patched:
        print(f"  + {f}")
    print("skipped (already patched):")
    for f in skipped:
        print(f"  = {f}")


if __name__ == "__main__":
    main()
