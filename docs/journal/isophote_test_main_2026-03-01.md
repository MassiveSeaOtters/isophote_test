---
date: 2026-03-01
repo: isophote_test
branch: main
tags:
  - journal
  - docs
  - inputs
  - huang2013
---

## Progress

- Reorganized `inputs/` into canonical subdirectories: `inputs/huang2013/`, `inputs/examples/`, and `inputs/demos/`.
- Moved the Huang2013 catalog, models, configs, and workflow scripts into the new canonical `inputs/huang2013/` layout.
- Moved example YAML files into `inputs/examples/` and moved demo scripts into `inputs/demos/`.
- Removed the temporary top-level compatibility wrappers and symlinks after validating the canonical script paths.
- Added a repo-local `AGENTS.md` and shortened `CLAUDE.md` to a compact project map.
- Rebuilt `docs/LESSON.md` from session notes and journals.
- Added `docs/todo.md` and `docs/plan/PLAN_DOC_REPO_TIDY.md` to track the cleanup work.
- Rewrote `inputs/README.md` as the canonical inventory and Huang2013 workflow table of contents.
- Updated active docs to the final canonical paths and current Huang2013 generator settings.
- Committed and fast-forward merged the work into `main` as `e197a77 Reorganize inputs and consolidate repo documentation`.

## Lessons Learned

- The canonical `inputs/` layout is clearer when Huang2013 assets, examples, and demos are separated physically rather than documented in one flat directory.
- Temporary compatibility layers are useful for validation, but they should be removed once the canonical paths are confirmed to avoid long-term documentation drift.
- The local `libprofit` runtime only worked when `LIBPROFIT_PATH` pointed to the `/Users/shuang/.../libprofit/mbp` path in this environment.
- Active docs need a second verification pass after structural cleanup to remove stale references to transitional paths.

## Key Issues

- `main` is ahead of `origin/main` by one commit after the fast-forward merge.
- `inputs/data/` remains untracked and was intentionally excluded from the commit.
- The shell configuration still exposed a `/Users/mac/.../libprofit/mbp` path, while the successful validation used `/Users/shuang/.../libprofit/mbp`.
