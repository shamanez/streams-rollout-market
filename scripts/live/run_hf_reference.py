"""Stage 2: HF transformers reference forward over the same (prompt + response) tokens.
Produces trainer-side logprobs for the rollout response tokens."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROLLOUT = json.loads(Path("/tmp/rollout.json").read_text())
# The trainer-side reference forward should use the bf16 (full-precision)
# checkpoint even when the rollout came from a quantized variant — the
# whole point of the cross-precision comparison is to measure how far the
# quantized rollout drifts from the trainer's logprob view.
MODEL = ROLLOUT.get("trainer_reference", ROLLOUT["model"])
prompt_ids = ROLLOUT["prompt_token_ids"]
response_ids = ROLLOUT["response_token_ids"]

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()
print(f"[hf] loaded in {time.time()-t0:.1f}s", flush=True)
print(f"[hf] device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}", flush=True)

P, R = len(prompt_ids), len(response_ids)
ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
ids_dev = ids.to(next(model.parameters()).device)
with torch.no_grad():
    out = model(input_ids=ids_dev)
logits = out.logits.float()  # [1, P+R, vocab]
# Token at position p is predicted by logits[p-1]; the response covers [P, P+R-1]
# so logits[P-1 .. P+R-2] are the relevant predictors.
trainer_logprobs = []
for i in range(R):
    logits_i = logits[0, P - 1 + i, :]
    log_probs = torch.log_softmax(logits_i, dim=-1)
    trainer_logprobs.append(float(log_probs[response_ids[i]].item()))
print(f"[hf] computed {len(trainer_logprobs)} trainer logprobs", flush=True)

Path("/tmp/trainer.json").write_text(json.dumps({
    "model": MODEL,
    "trainer_logprobs": trainer_logprobs,
    "engine": "hf-transformers",
    "attn_impl": "sdpa",
    "dtype": "bfloat16",
}))
print("[hf] wrote /tmp/trainer.json")
print(f"[hf] total {time.time()-t0:.1f}s")
