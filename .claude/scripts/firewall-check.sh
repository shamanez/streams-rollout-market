#!/usr/bin/env bash
# PreToolUse hook for Write/Edit: enforce AGENTS.md firewall.
# Blocks trainer code from being written to production modules.
set -uo pipefail

# Read tool input from stdin (hook receives JSON)
INPUT=$(cat)

# Trainer-related patterns that must not appear in production code
BLOCKED_PATTERNS="(class\s+(PPOTrainer|GRPOTrainer|RLOOTrainer|DPOTrainer)|def\s+backward|optimizer\.step|loss\.backward|gradient_step|fsdp_config|megatron_config|reward_model_training)"

# Allow if the target path is under tests/ or examples/
if echo "$INPUT" | grep -qE '"(file_path|path)"\s*:\s*"[^"]*(tests/|examples/)'; then
  exit 0
fi

# Check if the content contains blocked patterns
if echo "$INPUT" | grep -qiE "$BLOCKED_PATTERNS"; then
  cat <<'BLOCKED' >&2
BLOCKED by AGENTS.md firewall: This appears to be trainer code.
This repo does NOT implement PPO, GRPO, RLOO, DPO, FSDP, Megatron,
optimizer, gradient-step, or reward-model-training logic.
Trainer mocks are allowed only under tests/ or examples/.
BLOCKED
  exit 2
fi

# Block hardcoded API keys / secrets from being written to any file
SECRET_PATTERNS="(nvapi-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}|csk-[A-Za-z0-9]{20,}|sk-or-v1-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})"

# Allow secrets only in .env files (which are gitignored)
if echo "$INPUT" | grep -qE '"(file_path|path)"\s*:\s*"[^"]*\.env"'; then
  exit 0
fi

if echo "$INPUT" | grep -qE "$SECRET_PATTERNS"; then
  cat <<'BLOCKED' >&2
BLOCKED: Detected hardcoded API key or secret.
API keys must ONLY be stored in .env (which is gitignored).
Use os.environ["KEY_NAME"] or python-dotenv to load them in code.
BLOCKED
  exit 2
fi

exit 0
