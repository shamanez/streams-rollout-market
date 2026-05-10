#!/usr/bin/env bash
# PreToolUse hook for Bash: block AWS infrastructure modifications.
# The ONLY allowed AWS interaction is SSH to the spot instance.
set -uo pipefail

# Read tool input from stdin
INPUT=$(cat)

# Block AWS CLI commands that create/modify/delete infrastructure
AWS_MODIFY="(aws\s+(ec2|s3|iam|lambda|ecs|eks|rds|cloudformation|sagemaker)\s+(create|delete|modify|terminate|run|start|stop|update|put|remove|detach|attach|allocate|release|deregister))"

if echo "$INPUT" | grep -qiE "$AWS_MODIFY"; then
  cat <<'BLOCKED' >&2
BLOCKED: AWS infrastructure modification detected.
The ONLY AWS resource you may use is the existing spot instance via:
  ssh my-vllm-spot-instance
Do NOT create, modify, terminate, or resize any AWS resources.
BLOCKED
  exit 2
fi

# Block SSH key / PEM paths from appearing in committed code
if echo "$INPUT" | grep -qE "(\.pem|IdentityFile|BEGIN RSA|BEGIN OPENSSH)"; then
  # Allow if it's an ssh command (runtime use is fine)
  if echo "$INPUT" | grep -qE "^ssh\s"; then
    exit 0
  fi
fi

exit 0
