#!/usr/bin/env bash
# Creates or updates the fastapi-mcp-router label taxonomy (the label axes
# only). Types (Bug/Feature/Chore/Security/Idea) and Priority are native
# org-level GitHub fields, not labels, and are configured in org settings.
#
# The areas below are this repository's own, derived from its module
# boundaries. Keep them in sync with the glob map in .github/labeler.yml.
#
# One signal per axis; label text is the load-bearing distinction (WCAG 1.4.1).
# area:* uniform blue; on-hold gray (parked).
#
# Usage: .github/scripts/sync-labels.sh            (defaults to rcrsr/fastapi-mcp-router)
#        REPO=owner/name .github/scripts/sync-labels.sh
#
# Idempotent: `gh label create --force` upserts, so re-running only updates
# color/description drift. Requires: gh, authenticated with repo scope.
set -euo pipefail

REPO="${REPO:-rcrsr/fastapi-mcp-router}"

AREA_COLOR="1d76db"   # blue, uniform across every area
HOLD_COLOR="d2dae1"   # gray, parked/inactive

declare -a AREAS=(
  "area:router|MCP dispatch and transport: JSON-RPC, Streamable HTTP/SSE, auth, PRM, pagination"
  "area:registry|tool registration, schema generation, dependency injection, content blocks"
  "area:resources|resource registry, providers, URI templates, subscriptions"
  "area:prompts|prompt registry and argument metadata"
  "area:session|session stores, Redis, sampling, roots"
  "area:telemetry|optional OpenTelemetry tracing and metrics"
  "area:conformance|official MCP conformance suite harness and baseline"
  "area:docs|README, CHANGELOG, docs/, examples/"
  "area:dx|CI, toolchain, git hooks, root config"
)

for entry in "${AREAS[@]}"; do
  name="${entry%%|*}"
  desc="${entry#*|}"
  gh label create "$name" --repo "$REPO" --color "$AREA_COLOR" --description "$desc" --force
done

gh label create "on-hold" --repo "$REPO" --color "$HOLD_COLOR" \
  --description "Shaped work deliberately parked; not low priority, not blocked-by a specific issue" --force

echo "Label taxonomy synced to $REPO."
