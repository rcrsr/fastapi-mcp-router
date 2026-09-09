# Release SOP

Standard operating procedure for cutting a release of `fastapi-mcp-router`.
Consumed by `/conduct:cut-release {version}`; the rules here override that
skill's auto-detected defaults.

## Scope

Whole-repository release only. This repo has one package and one changelog.
The `{qualifier}` argument is not used.

## Prerequisites

- Clean working tree on `main`, fast-forwarded to `origin/main`.
- `gh` authenticated against `rcrsr/fastapi-mcp-router`.
- `CHANGELOG.md` `## [Unreleased]` section has at least one entry.

## Version sources (bump BOTH)

The version lives in two files. `tests/test_version.py` fails if they differ.

| File | Line | Format |
|------|------|--------|
| `pyproject.toml` | `[project]` `version = "X.Y.Z"` | plain semver, quoted |
| `fastapi_mcp_router/__init__.py` | `__version__ = "X.Y.Z"` | plain semver, quoted |

No `VERSION` file. No sync script. Edit both values directly; change nothing else.

Verify after bumping:

```bash
uv run pytest tests/test_version.py --no-cov -q
```

## Changelog

Single file: `CHANGELOG.md` (Keep a Changelog 1.1.0, bracketed headings).

1. Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` (date from `date +%F`).
2. Insert a fresh empty `## [Unreleased]` heading above it.
3. Do NOT add bottom link-reference definitions. This file uses inline PR links
   only and has no `[Unreleased]: ...` block.

## Naming

| Item | Scheme | Example |
|------|--------|---------|
| Branch | `release/{version}` | `release/0.4.0` |
| Commit | `release: {version}` | `release: 0.4.0` |
| PR title | `Release {version}` | `Release 0.4.0` |
| Squash subject | `release: {version} (#{pr})` | `release: 0.4.0 (#13)` |
| Tag | `v{version}`, annotated | `v0.4.0` |

Commit only `pyproject.toml`, `fastapi_mcp_router/__init__.py`, and
`CHANGELOG.md`. Never `git add .`.

## Merge gate

Squash-merge only after all three hold:

1. `mergeStateStatus == CLEAN`.
2. Zero unresolved review threads.
3. `ci.yml` checks all passed. Pending is not a pass.

## Tag push triggers a deploy

Pushing `v*` runs `.github/workflows/release.yml`, which:

1. Runs the full test suite.
2. Builds sdist and wheel.
3. Publishes to PyPI via trusted publishing (`skip-existing: true`).
4. Creates the GitHub release with `--generate-notes` and attaches `dist/*`.

Consequences for the skill:

- Treat the tag push as a production publish. Confirm before pushing.
- **Skip `gh release create`.** The workflow owns the GitHub release. Creating
  one manually collides with it.
- After the tag push, report the workflow run URL and stop.

## Post-release verification

```bash
gh run list --workflow=release.yml --limit 1
gh release view vX.Y.Z --json url,assets --jq '.url, (.assets[].name)'
```

Confirm the release page lists a `.whl` and a `.tar.gz` and that
`https://pypi.org/project/fastapi-mcp-router/X.Y.Z/` resolves.

## Rollback

- Tag pushed but PyPI publish failed: fix on `main`, re-run the workflow. PyPI
  rejects re-uploads of an existing version; a new patch version is required if
  the artifact reached PyPI.
- PR merged but tag not pushed: nothing is public yet. Revert the squash commit
  on `main` if needed, or push the tag once the issue is resolved.
