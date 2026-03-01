# TODO

## Phase Tracking

| Step ID | Phase | Task | Status | Notes |
|---|---|---|---|---|
| P1.1 | P1 | Create lean `AGENTS.md` | Done | Repo-local agent guidance added |
| P1.2 | P1 | Shorten `CLAUDE.md` | Done | Reduced to project map and references |
| P1.3 | P1 | Rebuild `docs/LESSON.md` | Done | Durable lessons extracted from session docs |
| P2.1 | P2 | Reorganize canonical `inputs/` assets into subdirectories | Done | Split into `huang2013`, `examples`, and `demos` |
| P2.2 | P2 | Add compatibility wrappers and symlinks for old paths | Done | Added during initial reorganization, then removed after validation |
| P2.3 | P2 | Make moved scripts path-safe | Done | Resource lookups now resolve from new locations |
| P3.1 | P3 | Rewrite `inputs/README.md` as a table of contents | Done | Canonical inventory and workflow documented |
| P3.2 | P3 | Fix stale path and parameter drift in related docs | Done | Active docs updated to canonical paths |
| P3.3 | P3 | Add plan document and review note | Done | See `docs/plan/PLAN_DOC_REPO_TIDY.md` |
| P4.1 | P4 | Inspect HSC sky catalogs and the `side1` downloader workflow | Done | `cosmos_dud_sky.csv` and `cosmos_wide_sky.csv` overlap at 223 exact coordinates |
| P4.2 | P4 | Add a flexible HSC cutout sky-statistics script | Done | Added `scripts/hsc_sky_calibration.py` with batch download, pooled sigma-clipped statistics, and QA outputs |
| P4.3 | P4 | Map measured HSC sky statistics into MockGal noise parameters | Done | `coadd/bg` pooled centered RMS maps to `sky_sb_limit`; original cutout medians are reported as exploratory sky-level summaries |

## Review

- Canonical `inputs/` paths now reflect actual usage.
- The temporary compatibility layer was removed after canonical path validation succeeded.
- Active docs were aligned to `docs/LESSON.md` and current Huang2013 mock settings.
- The HSC sky calibration workflow now centers each cutout locally and pools pixels by layer, matching the intended wide-vs-dud depth calibration logic.
- The repo-local script is designed around batch `side1` downloads with image and variance planes only, plus visual QA against synthetic MockGal-style noise patches.
