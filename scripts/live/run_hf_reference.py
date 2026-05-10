"""Stage 2: HF transformers reference forward over (prompt + response) tokens.

Reads /tmp/rollouts.json (a list of per-prompt dicts produced by
run_{vllm,sglang}_rollout.py), teacher-forces each one through the
trainer-reference checkpoint, and writes /tmp/trainers.json with the
matching list of trainer-side logprob arrays.

Single-prompt back-compat: if /tmp/rollouts.json is missing but
/tmp/rollout.json (the old flat schema) exists, the latter is wrapped
in a length-1 list. Output is always written to both /tmp/trainers.json
and /tmp/trainer.json (the latter being the first element).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/ubuntu/hf-cache")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_rollouts() -> list[dict]:
    path = Path("/tmp/rollouts.json")
    if path.exists():
        return json.loads(path.read_text())
    flat = json.loads(Path("/tmp/rollout.json").read_text())
    return [flat]


def main() -> None:
    rollouts = _load_rollouts()
    # Trainer-side reference: bf16 of the *original* checkpoint, even if
    # the rollout came from a quantized variant. trainer_reference is
    # constant across the whole batch (one reference model per run).
    model_id = rollouts[0].get("trainer_reference") or rollouts[0]["model"]

    t0 = time.time()
    AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[hf] loaded {model_id} in {time.time()-t0:.1f}s", flush=True)

    primary_dev = next(model.parameters()).device
    trainers = []
    t1 = time.time()
    for entry in rollouts:
        prompt_ids = entry["prompt_token_ids"]
        response_ids = entry["response_token_ids"]
        P, R = len(prompt_ids), len(response_ids)
        ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=primary_dev)
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits.float()
        trainer_logprobs = []
        for i in range(R):
            log_probs = torch.log_softmax(logits[0, P - 1 + i, :], dim=-1)
            trainer_logprobs.append(float(log_probs[response_ids[i]].item()))
        trainers.append({
            "model": model_id,
            "trainer_logprobs": trainer_logprobs,
            "engine": "hf-transformers",
            "attn_impl": "sdpa",
            "dtype": "bfloat16",
            "prompt_idx": entry.get("prompt_idx", 0),
        })
        print(f"[hf]   prompt {entry.get('prompt_idx', 0)}: {R} tokens", flush=True)
    print(f"[hf] forwarded {len(trainers)} prompt(s) in {time.time()-t1:.1f}s", flush=True)

    Path("/tmp/trainers.json").write_text(json.dumps(trainers))
    if trainers:
        Path("/tmp/trainer.json").write_text(json.dumps(trainers[0]))
    print(f"[hf] wrote /tmp/trainers.json ({len(trainers)} entries)")
    print(f"[hf] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
