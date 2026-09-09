#!/usr/bin/env bash
# Run the official MCP conformance suite against the fixture server.
#
# Usage:
#   bash conformance/run.sh                            # active suite, every spec version below
#   bash conformance/run.sh --scenario tools-list      # forward extra flags to the suite
#   bash conformance/run.sh --spec-version 2025-06-18  # one spec version only
#
# Env:
#   CONFORMANCE_PORT           port for the fixture server (default 3001)
#   CONFORMANCE_VERSION        pinned @modelcontextprotocol/conformance version (default 0.1.16)
#   CONFORMANCE_RESULTS        results dir (default conformance/results)
#   CONFORMANCE_SPEC_VERSIONS  space-separated spec revisions to run (default "2025-11-25 2025-06-18")
#
# 2025-03-26 is deliberately absent: conformance 0.1.16 ships zero server
# scenarios for it, so a run would exit 0 having tested nothing.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${CONFORMANCE_PORT:-3001}"
VERSION="${CONFORMANCE_VERSION:-0.1.16}"
RESULTS="${CONFORMANCE_RESULTS:-conformance/results}"
SPEC_VERSIONS="${CONFORMANCE_SPEC_VERSIONS:-2025-11-25 2025-06-18}"
URL="http://127.0.0.1:${PORT}/mcp"
BASELINE="conformance/baseline.yml"

rm -rf "$RESULTS"  # script-owned, gitignored; keep only the latest run
mkdir -p "$RESULTS"
uv run uvicorn conformance.app:app --host 127.0.0.1 --port "$PORT" --log-level warning \
  > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

server_ready=0
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null -X POST -H 'Content-Type: application/json' -H 'Accept: application/json' \
      -d '{"jsonrpc":"2.0","id":0,"method":"ping"}' "$URL" 2>/dev/null; then
    server_ready=1
    break
  fi
  sleep 0.25
done

if [ "$server_ready" -ne 1 ]; then
  echo "ERROR: fixture server did not become ready at $URL within 15s." >&2
  echo "See $RESULTS/server.log for details." >&2
  exit 1
fi

run_suite() {
  npx -y "@modelcontextprotocol/conformance@${VERSION}" server \
    --url "$URL" \
    --expected-failures "$BASELINE" \
    "$@"
}

# An explicit --spec-version wins over the default loop.
case " $* " in
  *" --spec-version "*) run_suite --output-dir "$RESULTS" "$@"; exit $? ;;
esac

status=0
for spec in $SPEC_VERSIONS; do
  echo "=== conformance: spec-version $spec ==="
  run_suite --spec-version "$spec" --output-dir "$RESULTS/$spec" "$@" || status=1
done
exit $status
