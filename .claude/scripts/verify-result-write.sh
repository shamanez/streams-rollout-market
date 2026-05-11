#!/usr/bin/env bash
# PreToolUse hook (matcher: Write|Edit) for the cwc-long-running-agents
# default-FAIL contract. Denies any Write/Edit that flips an entry in
# .claude/feature-results.json from `passes: false` to `passes: true`
# UNLESS the entry's `evidence` field points to a path that exists AND
# is referenced from .claude/evidence/evidence_log.jsonl.
#
# Exit 0 -> allow. Exit 2 -> deny (Claude sees stderr).

set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Read the hook payload from stdin into a temp file so the Python heredoc
# does not consume the same stdin.
PAYLOAD_FILE="$(mktemp -t verify-result-write.XXXXXX)"
trap 'rm -f "$PAYLOAD_FILE"' EXIT
cat > "$PAYLOAD_FILE"

ROOT="$ROOT" \
EVIDENCE_LOG="$ROOT/.claude/evidence/evidence_log.jsonl" \
RESULTS_REL=".claude/feature-results.json" \
PAYLOAD_FILE="$PAYLOAD_FILE" \
python3 <<'PY'
import json, os, re, sys

root = os.environ["ROOT"]
evidence_log = os.environ["EVIDENCE_LOG"]
results_rel = os.environ["RESULTS_REL"]
payload_file = os.environ["PAYLOAD_FILE"]

with open(payload_file, "r") as f:
    raw = f.read()

try:
    payload = json.loads(raw)
except Exception:
    sys.exit(0)

tool_name = payload.get("tool_name", "")
tool_input = payload.get("tool_input", {}) or {}
file_path = tool_input.get("file_path", "") or ""

if not file_path.endswith(results_rel):
    sys.exit(0)

if tool_name == "Write":
    fragment = tool_input.get("content", "") or ""
elif tool_name == "Edit":
    fragment = tool_input.get("new_string", "") or ""
else:
    sys.exit(0)

try:
    with open(evidence_log, "r") as f:
        refs = f.read()
except Exception:
    refs = ""

problems = []

def check_entry(key, evidence_value):
    if evidence_value is None or evidence_value == "" or evidence_value == "null":
        problems.append(f"evidence path required for {key} (got null/empty)")
        return
    ev = evidence_value
    ev_abs = ev if os.path.isabs(ev) else os.path.join(root, ev)
    if not os.path.exists(ev_abs):
        problems.append(f"evidence path does not exist for {key}: {ev}")
        return
    if os.path.basename(ev) not in refs and ev not in refs:
        problems.append(
            f"evidence path for {key} not referenced from evidence_log.jsonl: {ev}"
        )

parsed_full = None
try:
    parsed_full = json.loads(fragment)
except Exception:
    parsed_full = None

if isinstance(parsed_full, dict) and tool_name == "Write":
    for key, val in parsed_full.items():
        if isinstance(val, dict) and val.get("passes") is True:
            check_entry(key, val.get("evidence"))
else:
    pat = re.compile(
        r'"([A-Za-z0-9_.\-]+)"\s*:\s*\{[^{}]*?"passes"\s*:\s*true[^{}]*?\}',
        re.DOTALL,
    )
    ev_pat = re.compile(r'"evidence"\s*:\s*(null|"([^"]*)")')
    for m in pat.finditer(fragment):
        key = m.group(1)
        block = m.group(0)
        em = ev_pat.search(block)
        if em is None:
            problems.append(f"evidence field missing for {key}")
            continue
        if em.group(1) == "null":
            check_entry(key, None)
        else:
            check_entry(key, em.group(2))

if problems:
    for p in problems:
        sys.stderr.write(f"verify-result-write: {p}\n")
    sys.exit(2)

sys.exit(0)
PY
