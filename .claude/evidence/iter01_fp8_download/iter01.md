# Iter 01 (cycle FP8) — download Qwen/Qwen3-30B-A3B-FP8 to the spot

Part of the **MoE rollouts × precision matrix (bf16 + fp8) end-to-end**
initiative seeded 2026-05-13. Iter 1 of 8.

## Directive

Fetch the FP8 quantized MoE weights from HuggingFace into the spot's
`~/hf-cache/` and create a convenience symlink at
`~/checkpoint/qwen3-30b-a3b-fp8/hf`.

## What I did

1. Used `huggingface_hub.snapshot_download(repo_id="Qwen/Qwen3-30B-A3B-FP8",
   cache_dir="~/hf-cache/hub")` from inside `~/rmenv-dev/`.
2. After download, created the convenience symlink with
   `ln -sfn ~/hf-cache/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba73…/
   ~/checkpoint/qwen3-30b-a3b-fp8/hf` — same layout as the bf16 model.

## Acceptance evidence

```
$ ls -la ~/checkpoint/qwen3-30b-a3b-fp8/hf/config.json
lrwxrwxrwx … /home/ubuntu/checkpoint/qwen3-30b-a3b-fp8/hf/config.json ->
  ../../blobs/0048f472a9396c7a349f8c2136440553cb0b88c0
$ python3 -c 'import json; c = json.load(open("…/config.json")); \
              print(c["quantization_config"]["quant_method"],
                    c["quantization_config"]["fmt"])'
fp8 e4m3
```

Quantization config (truncated; full list of `modules_to_not_convert`
omitted):

```json
{
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "weight_block_size": [128, 128],
  "modules_to_not_convert": [
    "lm_head",
    "model.layers.0..47.{input_layernorm, mlp.gate, post_attention_layernorm}"
  ]
}
```

## Storage cost

| Model                       | Total size | Shards |
|-----------------------------|-----------:|-------:|
| Qwen/Qwen3-30B-A3B (bf16)   | ~60 GB     | 16     |
| Qwen/Qwen3-30B-A3B-FP8      | ~31 GB     | 7      |

FP8 is ~half the on-disk bytes of bf16, as expected (8-bit vs 16-bit
weights, with the routing gates left in higher precision).

## Notes for downstream iterations

* The bf16 and FP8 models share the *same* tokenizer (Qwen3 family),
  so trajectory `tokenizer_hash` will collapse — making bf16/FP8
  rollouts directly comparable.
* The MoE topology is identical: 48 layers × 128 experts each, top-8
  routing. The trainer-side teacher-force in bf16 will pick up the
  bf16 weight version regardless of which precision the rollout used.
* `quantization_config.modules_to_not_convert` keeps every layer's
  `mlp.gate` (the routing network) in high precision. So FP8 only
  affects expert *weights*, not routing decisions directly — any
  flip-rate change between bf16 and FP8 rollouts comes from FP8
  expert outputs propagating subtly different residual-stream
  activations into the next layer's gate.
