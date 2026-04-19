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
| P5.1 | P5 | Scope HSC `i`-band calibration outputs to the Huang2013 workflow | Done | Added `inputs/huang2013/configs/huang2013_hsc_i_calibration.yaml` as an opt-in reference file |
| P5.2 | P5 | Prepare a cleaner Huang2013 production-oriented layout | Done | Added `inputs/huang2013/README.md`, production output conventions, and generator run metadata |
| P5.3 | P5 | Scope calibrated Huang2013 depth values without changing repo defaults | Done | HSC `i`-band `wide` and `dud` values live in a Huang2013 reference file, not a repo-wide default |
| P6.1 | P6 | Replace the hardcoded Huang2013 batch script with a run-manifest workflow | Done | The production script now requires `--run-manifest` and resolves one row per mock configuration |
| P6.2 | P6 | Add canonical Huang2013 production manifests | Done | Added baseline, HSC `i`-band `wide`, and HSC `i`-band `dud` manifests under `inputs/huang2013/runs/` |
| P6.3 | P6 | Persist resolved production-run artifacts next to outputs | Done | The script now writes original/resolved manifests plus `run_metadata.json` into each output directory |
| P7.1 | P7 | Phase 1 investigation of GALFIT format, docs, and community parsers | Done | Web docs + README.pdf + `/Users/shuang/code/galfit/EXAMPLE.INPUT` reviewed; GALFITools/EllipSect/galfit-python-parser evaluated; design approved in `docs/plan/PLAN_GALFIT_PARSER.md` |
| P7.2 | P7 | Phase 2a: data model + tokenizer + value-envelope parser | Done | `FittedValue` handles `[]`/`{}`/`*`/`(err)` in any order |
| P7.3 | P7 | Phase 2b: header + profile read/write round-trip | Done | All 11 profiles covered; `sersic{1,2,3}` suffix; EXAMPLE.INPUT round-trips cleanly |
| P7.4 | P7 | Phase 2c: hidden blocks (Z, C0, B*, F*, R*, T*, Ti, To) | Done | Sparse indices; `R0` enum; truncation pseudo-components; `extra_params` absorbs GALFIT 3.0.7's reserved `6)/7)/8)` slots |
| P7.5 | P7 | Phase 2d: constraint file reader/writer (six line grammars) | Done | `galfit_constraints.py`; EXAMPLE.CONSTRAINTS round-trips |
| P7.6 | P7 | Phase 2e: YAML/JSON serialization + idempotent round-trip | Done | Both modules |
| P7.7 | P7 | Phase 2f: integration test against real `/Users/shuang/code/galfit/galfit` | Done | GALFIT 3.0.7 accepts parser-written config; parser reads its `galfit.01` |
| P7.8 | P7 | Phase 3: SKILL package for global install | Done | Installed at `~/.claude/skills/galfit/` (visible to Claude Code) and mirrored at `~/Dropbox/work/project/vibe/guangtou_vibe/skills/galfit/`; ships parser + CLI + 7 reference docs |
| P8.1 | P8 | CS4G plan + decision evidence (D4 i-band confirm, D5 prediction quality) | Done | Plan in `docs/plan/PLAN_CS4G_SAMPLE.md`; D4/D5 evidence in `.scratch/cs4g_d4_d5_evidence.png` |
| P8.2 | P8 | Build candidate sample with predicted M_i | Done | `inputs/cs4g/scripts/build_sample.py` → `cs4g_candidates.csv` (1642 rows) |
| P8.3 | P8 | Probe IRSA P4 directories, tag complexity | Done | `inputs/cs4g/scripts/probe_p4.py` → `cs4g_p4_index.csv` (1208 with rank ≥ 2) |
| P8.4 | P8 | Fetch best `.outgal` per galaxy with local mirror | Done | `inputs/cs4g/scripts/fetch_outgals.py` → `inputs/cs4g/p4/{name}/` |
| P8.5 | P8 | Parse outgals with /galfit skill, drop edgedisk galaxies | Done | `inputs/cs4g/scripts/parse_outgals.py` → `cs4g_models.json` (1198 kept, 10 edgedisk-skipped) |
| P8.6 | P8 | Magnitude/size conversion to mockgal schema | TODO | `inputs/cs4g/scripts/cs4g_to_mockgal.py` → `cs4g_components.json` |
| P8.7 | P8 | Stratified downsample to ~200 | TODO | `inputs/cs4g/scripts/downsample.py` → `cs4g_sample.csv` |
| P8.8 | P8 | MockGal manifest YAML | TODO | `inputs/cs4g/runs/cs4g_test.yaml` |
| P8.9 | P8 | Validation: render 5 vs S4G `_subcomps.fits` cubes | TODO | Eyeball check |
| P9.1 | P9 | Refactor mockgal.py to polymorphic Component ABC | Done | `Component` ABC with `to_libprofit_spec`, `to_astropy_image`, `derived_params`, `angular_extent_arcsec`; `RenderContext` bundles per-render env; `SersicEngine` aliased to `RenderEngine`; bit-identical parity with pre-refactor reference |
| P9.2 | P9 | Add `FerrerComponent` (libprofit only) | Done | 21 tests; astropy path raises `NotImplementedError`; registered under `type: ferrer`; defaults `alpha=2, beta=0` match Salo+2015 bar fits |
| P9.3 | P9 | Add `PointSourceComponent` with hard PSF guard | Done | 21 tests; stamps delta in astropy path; auto-size falls back to `MIN_IMAGE_EXTENT_PIX = 51` for PSF-only galaxies; registered under `type: psf`; raises `ValueError` if `psf_enabled=False` |
| P9.4 | P9 | Add `mockgal_galfit.py` (GALFIT-backed reference renderer) | Done | Wraps GALFIT 3.0.7 binary in `P) 1` model-only mode via the bundled `/galfit` skill's `write_galfit`; two input modes (MockGalaxy + GALFIT-dict-direct); 13 tests |
| P9.5 | P9 | Component-ABC contract test | Done | Walks `_COMPONENT_REGISTRY`, asserts the four abstract methods are implemented and that each registered class instantiates; catches future profile types added without proper interface compliance |

## Review

- Canonical `inputs/` paths now reflect actual usage.
- The temporary compatibility layer was removed after canonical path validation succeeded.
- Active docs were aligned to `docs/LESSON.md` and current Huang2013 mock settings.
- The HSC sky calibration workflow now centers each cutout locally and pools pixels by layer, matching the intended wide-vs-dud depth calibration logic.
- The repo-local script is designed around batch `side1` downloads with image and variance planes only, plus visual QA against synthetic MockGal-style noise patches.
- The Huang2013 workflow now documents HSC `i`-band `wide` and `dud` depth references without turning them into repo-wide defaults.
- The Huang2013 production workflow is now manifest-driven, which makes adding or revising mock configurations a data change rather than a code change.
- Each production output directory now stores the original manifest, the fully resolved manifest, and run metadata for reproducibility.
- Huang2013 manifests can now record `max_image_size` explicitly, and the baseline run uses a low-noise reference row instead of a zero-noise row.
- The Huang2013 model file now stores `re_overall`, and production sizing is anchored on that galaxy-level radius with `size_factor = 6`.
- Focused verification is clean again after restoring the `load_model_file` test import, teaching `engine=\"auto\"` to ignore unusable `profit-cli` binaries, and fixing the visualization colorbar label for LaTeX-backed matplotlib sessions.
- mockgal now renders multi-component galaxies (Sersic + Ferrer + PSF-convolved point source) via a polymorphic `Component` ABC. Adding new profile types requires implementing four methods and registering in `_COMPONENT_REGISTRY` rather than touching three dispersed dispatch sites. The refactor is behavior-preserving: bit-identical render output on the existing single-Sersic and multi-Sersic reference cases.
- Pixel-perfect GALFIT renders are available via `mockgal_galfit.py`, which wraps the GALFIT binary in model-only mode and shares the bundled `/galfit` skill's writer. Use it as a reference renderer when validating mockgal's libprofit/astropy paths against the original GALFIT integrators.
