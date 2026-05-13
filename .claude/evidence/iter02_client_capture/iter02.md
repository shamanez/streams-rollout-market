# Iter 02 (cycle Remote-vLLM) — client-side probe-capture driver

## Directive

Write `scripts/live/remote_capture_driver.py` — same OpenAI tool-
using loop as `minimal_agent_driver.py` but extracts probes from
the response JSON client-side, no proxy required. Sidecar shape
must match `extract_probes`'s output byte-for-byte so downstream
merger code is unaware of the switch.

## What I did

1. **`scripts/live/remote_capture_driver.py`** — imports the
   shared helpers from sibling modules (no code duplication):
   - `inject_logprobs`, `extract_probes`, `derive_turn_idx` from
     `logprob_capture_proxy.py` — same request augmentation and
     same response-side parse as the proxy.
   - `TOOLS_OPENAI_SPEC`, `_dispatch_tool`, `_conv_from_messages`
     from `minimal_agent_driver.py` — same tool surface and
     ShareGPT serialization.

   The only new code is `_chat_once_with_capture`: it serializes
   the request body, runs it through `inject_logprobs`, POSTs the
   augmented bytes, captures the raw response, calls
   `extract_probes(req_bytes, resp_bytes, prompt_index=..., turn_idx=...)`,
   and appends one JSONL line to `--sidecar`. The returned dict is
   the parsed response so the caller's tool loop doesn't need any
   changes.

   Module-level `sys.path.insert` makes the two sibling imports
   work without turning `scripts/live/` into a package.

2. **`tests/test_remote_capture_driver.py`** — two tests:
   - `test_driver_captures_same_fields_as_proxy`: monkey-patches
     `requests.post` to return a synthetic chat-completions
     response with all four probe fields populated. Verifies (a)
     the bytes the driver POSTs match `inject_logprobs(raw)`
     exactly, (b) the sidecar line equals `extract_probes(...)`
     output field-for-field (modulo the `timestamp` the driver
     adds for parity with the proxy), and (c) the parsed JSON
     comes back to the caller intact.
   - `test_driver_skips_sidecar_write_when_response_has_no_logprobs`:
     models the "operator forgot to apply the patch" path — the
     driver still returns the response so the tool loop survives
     but writes nothing to the sidecar.

## Acceptance evidence

```text
$ pytest -q tests/test_remote_capture_driver.py
..                                                                       [100%]
2 passed in 0.11s

$ pytest -q | tail -3
.....................................                                   [100%]
473 passed in 2.66s

$ ruff check .
All checks passed!
```

Acceptance criterion ("unit test verifies sidecar JSONL shape
matches `extract_probes`'s output schema field-for-field") is met
by `test_driver_captures_same_fields_as_proxy`: the assertion
`driver_record_no_ts == expected_record` compares the driver's
record dict against `extract_probes`'s output dict for the same
inputs.

## Notes

* The driver's `--sidecar` default is `/tmp/hermes_probes.jsonl` —
  the same path the proxy uses. Downstream
  `merge_probes_into_trajectories.py` works without changes.
* The driver appends to the sidecar (just like the proxy). The
  caller is expected to `rm -f /tmp/hermes_probes.jsonl` before
  starting a new batch if they want clean separation.
* No live HTTP traffic in either test — the monkey-patched
  `requests.post` returns a fixture. Iter 3 will run it against the
  real spot via SSH port-forward.
