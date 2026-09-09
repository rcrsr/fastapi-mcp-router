# MCP conformance suite

Runs the official `@modelcontextprotocol/conformance` server scenarios against
a fixture app built only from this library's public API.

```bash
bash conformance/run.sh                          # full active suite
bash conformance/run.sh --scenario tools-list    # one scenario
bash conformance/run.sh --suite all              # include pending scenarios
```

Requires Node.js 20+ (`npx`). Results land in `conformance/results/` (gitignored).

| File | Purpose |
|------|---------|
| `app.py` | Fixture server: tools, resources, prompts named by the scenarios |
| `run.sh` | Starts uvicorn on port 3001, runs the suite, tears down |
| `baseline.yml` | Expected failures, one comment per library gap |

The suite exits 1 on any scenario that fails outside the baseline, and also on
any baselined scenario that starts passing. When you close a gap, delete its
entry from `baseline.yml` in the same PR.

CI runs this in the `conformance` job. The `pre-push` lefthook gate runs it too.
