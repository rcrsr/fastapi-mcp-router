#!/usr/bin/env bash
# Run the official MCP conformance suite against the fixture server.
#
# Usage:
#   bash conformance/run.sh                       # active suite, 2025-11-25 spec version
#   bash conformance/run.sh --scenario tools-list # forward extra flags to the suite
#
# Env:
#   CONFORMANCE_PORT     port for the fixture server (default 3001)
#   CONFORMANCE_VERSION  pinned @modelcontextprotocol/conformance version (default 0.1.16)
#   CONFORMANCE_RESULTS  results dir (default conformance/results)
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${CONFORMANCE_PORT:-3001}"
VERSION="${CONFORMANCE_VERSION:-0.1.16}"
RESULTS="${CONFORMANCE_RESULTS:-conformance/results}"
URL="http://127.0.0.1:${PORT}/mcp"
BASELINE="conformance/baseline.yml"

mkdir -p "$RESULTS"
uv run uvicorn conformance.app:app --host 127.0.0.1 --port "$PORT" --log-level warning \
  > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null -X POST -H 'Content-Type: application/json' -H 'Accept: application/json' \
      -d '{"jsonrpc":"2.0","id":0,"method":"ping"}' "$URL" 2>/dev/null; then
    break
  fi
  sleep 0.25
done

npx -y "@modelcontextprotocol/conformance@${VERSION}" server \
  --url "$URL" \
  --spec-version 2025-11-25 \
  --expected-failures "$BASELINE" \
  --output-dir "$RESULTS" \
  "$@"
