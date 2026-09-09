# MCP conformance suite

Runs the official `@modelcontextprotocol/conformance` server scenarios against
a fixture app built only from this library's public API.

```bash
bash conformance/run.sh                            # active suite, spec 2025-11-25 and 2025-06-18
bash conformance/run.sh --spec-version 2025-06-18  # one spec revision
bash conformance/run.sh --scenario tools-list      # one scenario
bash conformance/run.sh --suite all                # include pending scenarios
```

Each spec revision pins the client to that `protocolVersion` and runs only
the scenarios tagged for it, so the older run exercises the version-clamping
path. `2025-03-26` is skipped on purpose: conformance 0.1.16 has no server
scenarios for it and would exit 0 having tested nothing.

Requires Node.js 20+ (`npx`). Results land in `conformance/results/` (gitignored).

| File | Purpose |
|------|---------|
| `app.py` | Fixture server: tools, resources, prompts named by the scenarios |
| `run.sh` | Starts uvicorn on port 3001, runs the suite, tears down |
| `baseline.yml` | Expected failures, one comment per library gap |

The suite exits 1 on any scenario that fails outside the baseline, and also on
any baselined scenario that starts passing. When you close a gap, delete its
entry from `baseline.yml` in the same PR.

CI runs this in the `conformance` job, one matrix entry per spec revision. The `pre-push` lefthook gate runs it too.
