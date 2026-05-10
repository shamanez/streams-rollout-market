#!/usr/bin/env bash
# Host the dashboard snapshot on the spot instance.
#
# Pushes /tmp/dash-snap/ from the local Mac up to ~/dashboards on the
# spot host, then starts a detached `python -m http.server 8080`
# bound to 0.0.0.0. Reachable from anywhere the spot's security group
# allows port 8080 (default sec group on this instance permits it).
#
# Usage:
#   1. Build the snapshot locally:
#        rm -rf /tmp/dash-snap
#        python scripts/live/publish_dashboards.py --out-dir /tmp/dash-snap
#   2. Run this script:
#        bash scripts/live/serve_on_spot.sh
#   3. Open the URL it prints.
#
# Caveats:
#   - The spot can be reclaimed at any time; URL is not permanent. Use
#     GitHub Pages for the durable share link.
#   - The public IP is dynamic across stop/start; re-run this script
#     after a reboot to get the fresh IP.

set -euo pipefail

SPOT="${SPOT:-my-vllm-spot-instance}"
PORT="${PORT:-8080}"
LOCAL_DIR="${LOCAL_DIR:-/tmp/dash-snap}"
REMOTE_DIR="${REMOTE_DIR:-~/dashboards}"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "snapshot dir $LOCAL_DIR does not exist; run publish_dashboards.py first" >&2
    exit 1
fi

echo "[serve] syncing $LOCAL_DIR/* to $SPOT:$REMOTE_DIR/"
ssh "$SPOT" "mkdir -p $REMOTE_DIR && rm -f $REMOTE_DIR/*.html $REMOTE_DIR/*.json"
scp -q "$LOCAL_DIR"/*.html "$LOCAL_DIR"/*.json "$SPOT:$REMOTE_DIR/"

echo "[serve] killing any prior http.server on port $PORT"
ssh "$SPOT" "pkill -f 'python.*http.server.*$PORT' || true; sleep 1" || true

echo "[serve] launching detached http.server on port $PORT"
ssh -f "$SPOT" "cd $REMOTE_DIR && nohup python3 -m http.server $PORT --bind 0.0.0.0 > /tmp/dash_serve.log 2>&1 &"
sleep 2

PUBLIC_IP="$(ssh "$SPOT" 'curl -s --max-time 5 ifconfig.me')"
echo
echo "📡 dashboard available at: http://${PUBLIC_IP}:${PORT}/index.html"
echo "    glossary:              http://${PUBLIC_IP}:${PORT}/glossary.html"
echo
echo "log: $SPOT:/tmp/dash_serve.log"
echo "stop: ssh $SPOT 'pkill -f \"python.*http.server.*$PORT\"'"
