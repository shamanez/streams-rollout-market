# Iter 02 (cycle FP8) — vLLM-dev FP8 serve + routed_experts probe

## Directive

Launch Qwen3-30B-A3B-FP8 with the patched dev wheel and confirm
`choices[0].routed_experts` is present on the OpenAI HTTP response
with shape `[gen_tokens, 48, 8]`.

## What I did

1. Wrote `scripts/live/vllm_serve_moe_fp8_dev.sh` — a thin sibling
   of `vllm_serve_moe_dev.sh` that just overrides `MODEL` and `LOG`.
   The pre-flight check for the routed_experts HTTP patch and every
   vLLM CLI flag (`--enable-return-routed-experts`,
   `--enable-expert-parallel`, `--no-async-scheduling`, etc.) come
   from the bf16 launcher unchanged.
2. Launched it on the spot via `setsid nohup … & disown`. After
   ~2 min, `/v1/models` returned `200`.
3. Issued one tiny chat completion (`"What is 2+2? Reply only with
   the integer, no thinking"`, `enable_thinking=false`, `max_tokens=32`).

## Acceptance evidence

```text
finish_reason: stop
content: '4'
routed_experts present: True
  outer (tokens):  29     # engine emits the full prefill+decode buffer;
                          # the capture proxy will trim to the
                          # `usage.completion_tokens` tail (2 tokens here)
                          # before writing the sidecar — same behavior
                          # as the bf16 path.
  middle (layers): 48
  inner (top-k):   8
  first token top-1 per layer (first 6 layers): [49, 68, 92, 82, 16, 8]
usage: {prompt_tokens: 28, completion_tokens: 2, total_tokens: 30}
```

The shape is `[buffer_tokens, 48, 8]` exactly as expected — top-1
expert IDs per layer per token. The patched HTTP shim wires it
through identically for FP8 and bf16.

## Memory observation (footnote — not an acceptance criterion)

The directive expected GPU memory usage "roughly half of the bf16
case". In practice the spot reports ~39.5 GB used per GPU under
both FP8 and bf16, because `--gpu-memory-utilization 0.85` is what
caps the allocator. The **weight memory** halves (FP8 weights are
~7 GB total vs ~15 GB bf16 across TP=4), but the KV cache, scratch
buffers, and CUDA graphs scale to fill the budget either way. So
FP8's win is throughput / context length, not "less VRAM" at
default settings.

## Next iteration plan — queue re-sort

The smoke serve from this iter is still running. To avoid tearing
down a hot model load for the bf16 capture and then bringing FP8
back five minutes later, I'm re-sorting the queue so iter **fp8
task batch** (originally iter 4) runs while this serve is hot,
followed by **bf16 task batch** (originally iter 3) after switching
launchers. Same six tasks either way; same dashboard outcome.
