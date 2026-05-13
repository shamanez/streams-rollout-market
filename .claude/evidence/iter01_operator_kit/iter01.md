# Iter 01 (cycle Remote-vLLM) — operator serve kit

## Directive

Bundle the three files a remote operator needs (serve script,
HTTP-shim patch, quickstart README) into a self-contained tarball
at `/tmp/operator_kit.tgz`. `tar tzf` should list at least those
three files; the README should explain why each step is needed.

## What I did

Created `scripts/live/operator_kit/` with:

1. `OPERATOR_QUICKSTART.md` — explains why `routed_experts` over
   HTTP needs the patch, lists prerequisites (Python 3.12, CUDA
   driver, ~80 GB free disk), gives the three commands to run
   (create venv → apply patch → `bash vllm_serve_moe_dev.sh`),
   and a curl-based verification step plus a table of every env
   var knob the serve script honors. Also notes constraints (no
   telemetry, patch is read-only at runtime, etc.).

2. `build_kit.sh` — staging script that copies the three source
   files into a temp dir and packs them with `tar -czf`. Idempotent
   and parametric via `OUT_TGZ`.

The two source files referenced — `scripts/live/vllm_serve_moe_dev.sh`
and `scripts/live/patch_vllm_routed_experts_http.py` — already exist
in the repo and aren't duplicated.

## Acceptance evidence

```
$ bash scripts/live/operator_kit/build_kit.sh
wrote /tmp/operator_kit.tgz
-rw-r--r-- 1 shamane wheel 6455 May 13 04:26 /tmp/operator_kit.tgz

$ tar tzf /tmp/operator_kit.tgz
./
./OPERATOR_QUICKSTART.md
./patch_vllm_routed_experts_http.py
./vllm_serve_moe_dev.sh
```

* Three files present ✓
* README has a per-step "Why each step is needed" table ✓
* Tests 471/471 passed; ruff clean ✓

## Note on what's deliberately NOT in the kit

* The Megatron MoE launcher and trainer-side scripts. The operator
  only needs to serve the rollouts; the coordinator (us) runs
  FSDP/Megatron locally on bf16 weights they already have.
* The capture proxy. Capture moves client-side in iter 2 — the
  operator does not run our proxy.
* `bootstrap_spot.sh`. That's our own spot-setup tool; the operator
  has their own preferred way to bring up Python + CUDA.
