#!/usr/bin/env bash
# Build a self-contained operator kit tarball.
#
# Output: /tmp/operator_kit.tgz (or $OUT_TGZ if set), containing
# everything a third-party GPU operator needs to start serving
# Qwen3-30B-A3B with the routed_experts HTTP shim enabled.
#
# Contents:
#   OPERATOR_QUICKSTART.md           prerequisites + 3 commands + verification
#   vllm_serve_moe_dev.sh            the launcher (with HTTP-shim sanity check)
#   patch_vllm_routed_experts_http.py    the 4-file surgical patch
#
# Run on the laptop:
#   bash scripts/live/operator_kit/build_kit.sh
#
# Then hand the tarball to the operator however you onboard them.

set -euo pipefail

OUT_TGZ="${OUT_TGZ:-/tmp/operator_kit.tgz}"
REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
STAGE="$(mktemp -d)"
trap "rm -rf $STAGE" EXIT

cp "$REPO_ROOT/scripts/live/operator_kit/OPERATOR_QUICKSTART.md" "$STAGE/"
cp "$REPO_ROOT/scripts/live/vllm_serve_moe_dev.sh" "$STAGE/"
cp "$REPO_ROOT/scripts/live/patch_vllm_routed_experts_http.py" "$STAGE/"

# Sanity: every file listed in the README must be in the tarball.
for f in OPERATOR_QUICKSTART.md vllm_serve_moe_dev.sh patch_vllm_routed_experts_http.py; do
  if [[ ! -f "$STAGE/$f" ]]; then
    echo "FATAL: staged $STAGE/$f missing" >&2
    exit 2
  fi
done

# Strip the dev-venv pre-flight assertion from the launcher since
# the operator's venv name might differ; the README explains the
# alternative env vars they can set. (The pre-flight check still
# runs on our spot via the same script — we just relax the
# assertion here for the operator-facing copy.)
# (Currently the launcher already honors $RMENV_DEV via env var, so
# no edits are needed. We leave the file as-is.)

tar -czf "$OUT_TGZ" -C "$STAGE" .
echo "wrote $OUT_TGZ"
ls -la "$OUT_TGZ"
echo "---"
echo "Contents:"
tar -tzf "$OUT_TGZ"
