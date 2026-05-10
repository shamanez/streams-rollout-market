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

exit 0
