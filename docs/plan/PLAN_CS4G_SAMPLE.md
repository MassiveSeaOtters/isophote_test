# PLAN: CS4G Mock-Galaxy Sample Preparation

## Goal

Build a structured, downsampled CS4G/S4G sample of complex late-type disk
galaxies with parsed multi-component GALFIT models, ready to drive
`mockgal` for systematic isophote-fitting tests. Uses the IRSA `cs4gcat`
catalog and the IRSA-hosted Salo+2015 P4 GALFIT outputs, parsed via the
local `/galfit` skill.

## Inputs

- `inputs/cs4g/cs4g_catalog.fits` — IRSA `cs4gcat` (3239 rows, 36 cols),
  has `mabs3` (i-band M_abs, AB) for the dwarf-galaxy sub-sample only.
- `inputs/cs4g/cs4g_2025.fit` — VizieR J/A+A/697/A38 (3239 rows, 137 cols),
  adds inclination, Hubble type strings, B-band photometry, kinematics.
- `inputs/cs4g/cs4g_catalog.tbl` — IPAC plain-text version with column
  descriptions; the authoritative source for column semantics.
- IRSA P4 archive: `https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{name}/P4/`
  serves Salo+2015 GALFIT outputs as `.outgal` ASCII files.

## Decisions (fixed)

| ID | Decision | Value |
|----|----------|-------|
| D1 | Late-type T-type cutoff | `cvrhs_t >= -3` (S0- and later) |
| D2 | Edge-on cutoff | `Incl < 70 deg` |
| D3 | Final sample size | aim for 200, smaller acceptable |
| D4 | Confirm `mabs3` band | i-band AB (3 lines of evidence; see `docs/lessons.md`) |
| D5 | M_i derivation | predicted for everyone (real `mabs3` is dwarf-only) |
| D6 | Color predictor | per-galaxy from `(B - 3.6μm)`, fallback constant |
| D7 | Most-complex selection | filename heuristic (no parse-then-count) |
| D8 | Local mirror | yes, under `inputs/cs4g/p4/{name}/` |
| D9 | Output band | i-band only |

## Joint cuts as of plan time

`T >= -3 & Incl < 70 & isfinite(mabs1) & isfinite(logMstar)` → **1642 candidates**.
Of those, ~10–15% expected to be 404 in IRSA P4 (no GALFIT model published).

## M_i prediction recipe (D5 + D6)

For galaxy with apparent IRAC1 mag `m1_app` at distance `D` (Mpc):

1. Distance modulus: `DM = 5 log10(D * 1e6) - 5`
2. M_3.6 = `m1_app - DM` (apparent → absolute)
3. Color (M_3.6 − M_i):
   - **If `Btot` available**: `color = -0.235 * (B_abs - M_3.6) + 0.670`
     (linear fit calibrated on N=366 dwarfs, RMS = 0.14 mag,
     CV held-out RMS 0.10–0.17 mag)
   - **Else**: `color = +0.573` (median of training set, RMS = 0.19 mag)
4. M_i = M_3.6 − color

The 366 calibrators are dwarfs (DG sub-sample). For S4G spirals the
0.14–0.19 mag scatter is acceptable; the catalog spans 6+ mag in M_i.

## Phases

### Phase 1 — Build candidate sample (`inputs/cs4g/build_sample.py`)

- Cross-join IRSA + VizieR on object name.
- Apply T, Incl, IRAC1, logMstar finite cuts.
- Compute predicted M_i per row (Strategy D / fallback A).
- Tag each row with `mi_source ∈ {real, predicted_B, predicted_const}`.
- Output: `inputs/cs4g/cs4g_candidates.csv` (1642 rows).

### Phase 2 — Probe P4 directories (`inputs/cs4g/probe_p4.py`)

- For each candidate, fetch the IRSA P4 directory listing
  (parallel, ~10 concurrent, polite rate-limit).
- Parse links, extract `.outgal` filenames.
- Tag each galaxy with `complexity_rank`:
  - 0 = no P4 entry (404)
  - 1 = `_onecomp.outgal` only
  - 2 = `_twocomp.outgal` present
  - 3 = `_twocomp.outgal` plus extra-tag (`_dbar`, `_ddbar`, etc.)
- Output: `inputs/cs4g/p4_index.csv` (one row per galaxy).

### Phase 3 — Fetch best `.outgal` per galaxy (`inputs/cs4g/fetch_outgals.py`)

- For each galaxy with `complexity_rank >= 1`, choose the most-complex
  `.outgal` by filename heuristic (see Phase 2).
- Download to `inputs/cs4g/p4/{name}/{filename}`.
- Skip already-cached files; re-fetch on `--force`.

### Phase 4 — Parse and component-tag (`inputs/cs4g/parse_outgals.py`)

- Use the bundled `/galfit` skill parser.
- Per-component, record: profile type, x/y, integrated mag (col `3)`),
  Re/h_s/etc. (col `4)`), n (col `5)` for sersic), q (col `9)`), PA
  (col `10)`), `# STRUCTURE:` label.
- For `edgedisk` components: drop the entire galaxy (mu0 path + libprofit
  unsupported). Log to a skip-list.
- Output: `inputs/cs4g/cs4g_models.json` keyed by galaxy name.

### Phase 5 — Magnitude/size conversion (`inputs/cs4g/cs4g_to_mockgal.py`)

- Per component, convert IRAC1 apparent mag → predicted M_i using
  per-galaxy color from Phase 1.
- Convert pixel sizes → arcsec via the GALFIT `K)` plate scale, then
  arcsec → kpc via the catalog distance.
- Map per-component {q, PA} into the `mockgal.SersicComponent` schema
  (axis ratio convention: q = b/a; PA convention: matches mockgal).
- Output: `inputs/cs4g/cs4g_components.json` (mockgal-ready records).

### Phase 6 — Downsample (`inputs/cs4g/downsample.py`)

- Stratify by `complexity_rank`: prefer rank 3 > rank 2 > rank 1.
- Within each stratum, KS-match `logMstar` distribution to the parent
  candidate sample.
- Target N = 200, fall back to whatever rank-3+rank-2 yields if smaller.
- Output: `inputs/cs4g/cs4g_sample.csv`.

### Phase 7 — MockGal manifest (`inputs/cs4g/runs/cs4g_test.yaml`)

- One YAML entry per sampled galaxy, mirroring Huang2013 conventions.
- Reference the per-galaxy model JSON from Phase 5.

### Phase 8 — Validation

- Render 5 galaxies with `mockgal`, eyeball against the IRSA-hosted
  `*_subcomps.fits` cube for the same galaxy.
- Sanity check on flux conservation, Re consistency.

## Risks & open issues

- **Color extrapolation**: the M_i predictor is calibrated on dwarfs but
  applied to spirals. Expected systematic error: ~0.1–0.2 mag. Fine for
  systematic isophote-fit tests but worth flagging in the output.
- **GALFIT mu0 columns on non-edgedisk profiles**: the GALFIT manual is
  inconsistent — a few profile types (e.g. `nuker`) use surface brightness
  at break radius. We expect to encounter only `sersic`, `expdisk`,
  `devauc`, `psf` in the S4G P4 catalog, but should error loudly if
  anything else appears.
- **Salo+2015 component conventions**: `# STRUCTURE:` labels are
  S4G-pipeline-specific, not standard GALFIT. Treat as advisory.
- **404 rate**: ETG sub-sample (456 gals) likely has high 404 rate since
  the S4G P4 catalog originally targeted late-type disks.
- **IRSA rate limiting**: be polite — cap concurrency at 10, add
  user-agent identifying the script.
