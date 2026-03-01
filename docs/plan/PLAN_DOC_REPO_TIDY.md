# PLAN_DOC_REPO_TIDY

## Summary

Tidy the repo documentation and `inputs/` tree so Huang2013 support is easy to locate, agent guidance is concise, and stale path drift is removed without breaking old entry points.

## Phase P1: Agent Docs And Lessons

### P1.1 Create repo-local `AGENTS.md`
Acceptance criteria:
- Repo-specific rules are concise and durable.
- `docs/LESSON.md` is referenced as the lessons source of truth.

### P1.2 Shorten `CLAUDE.md`
Acceptance criteria:
- It contains only a compact project map and references.
- It no longer duplicates lessons, troubleshooting, or long command lists.

### P1.3 Rebuild `docs/LESSON.md`
Acceptance criteria:
- Durable lessons are extracted from session notes and journals.
- Ephemeral branch and commit status is excluded.

## Phase P2: `inputs/` Reorganization

### P2.1 Move canonical assets into subdirectories
Acceptance criteria:
- `inputs/huang2013/`, `inputs/examples/`, and `inputs/demos/` each have a clear role.

### P2.2 Preserve old flat paths
Acceptance criteria:
- Old script entry points still work through wrappers.
- Old YAML and TXT paths still resolve through symlinks.

### P2.3 Fix moved-script path handling
Acceptance criteria:
- Scripts resolve repo-root and data paths from their new locations.
- Huang2013 help commands work from both old and new paths.

## Phase P3: Documentation Alignment

### P3.1 Rewrite `inputs/README.md`
Acceptance criteria:
- It serves as the canonical table of contents for all `inputs/` assets.
- The minimum Huang2013 test-suite files are obvious.

### P3.2 Fix drift in related docs
Acceptance criteria:
- Active docs use canonical `inputs/` paths.
- Active docs match current Huang2013 generator settings.

### P3.3 Close out tracking
Acceptance criteria:
- `docs/todo.md` reflects completed work.
- Validation results and residual risks are captured in the implementation summary.
