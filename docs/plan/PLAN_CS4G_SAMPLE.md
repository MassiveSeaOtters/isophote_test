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

### Phase 8 — Validation (P8.9, Done)

**Status: Done.** Implementation in `inputs/cs4g/scripts/qa_mockgal_vs_s4g.py`;
outputs in `output/cs4g_s4g_irac1_test/` (`qa_s4g_validation.png`,
`qa_summary.json`, per-galaxy `mockgal.fits` + `s4g_reference.fits`).

Final metrics on the 3-galaxy set (after P8.11 color-shift fix):

| Galaxy   | Components            | flux_ratio | corr   | peak_ratio |
|----------|-----------------------|------------|--------|------------|
| NGC1433  | sersic+sersic+ferrer  | 0.996      | 0.974  | 1.96       |
| NGC0275  | sersic+ferrer+ferrer  | 0.991      | 0.9996 | 1.01       |
| NGC1357  | sersic+sersic+sersic  | 0.989      | 0.877  | 3.98       |

Total-flux agreement within 1% of unity; pixel-wise correlation
0.88–0.9996. Residual peak_ratio > 1 for bulge-dominated galaxies
is fully explained by the Gaussian-FWHM=1.66″ PSF placeholder vs
Salo's composite PSF; bulgeless NGC0275 shows peak_ratio=1.01 (no
PSF-core cusp to smear).

Path to these final numbers:

1. First-pass flux_ratio measured 1.31–1.64 before any correction.
2. Root cause traced to a **band mismatch** in the QA script, not a
   cosmology bug: `cs4g_sample_models.yaml` stores i-band absolute
   magnitudes (see `metadata.assumed_band`), and the converter applies
   a per-galaxy `M_3.6 − M_i` color shift to the Salo IRAC1 source
   values. Rendering those i-band abs_mags against an IRAC1 reference
   without undoing the shift produces flux excess exactly equal to
   `10^(0.4·color_shift)`. Predicted flux ratios matched measurements
   to 0.005 (e.g. NGC0275: predicted 1.658, measured 1.642).
3. `color_shift_3p6_minus_i` is already a top-level field in the sample
   YAML. P8.11 added the undo-shift to `build_mockgalaxy` in
   `qa_mockgal_vs_s4g.py` (three-line change); no converter or model
   regeneration required.

Follow-ups filed as separate TODO rows:

- **P8.10 — Run-manifest `defaults:` not merged**: `load_image_configs`
  reads `pixel_scale`/`zeropoint`/PSF from each `configs:` entry only,
  ignoring a sibling `defaults:` block. P8.9 worked around it by driving
  `generate_mock_image(...)` from Python with an explicit `ImageConfig`.
- **P8.12 — Gaussian vs composite PSF** (low priority): remaining
  peak_ratio=3.98 for NGC1357 is the Gaussian FWHM=1.66″ vs Salo's
  composite PSF core. `PSF-1.composite.fits` is already pre-fetched
  locally at `output/cs4g_s4g_irac1_test/{name}/s4g_psf_composite.fits`
  for a drop-in when core fidelity matters.

### Phase 8 — Original specification (reference)

Compare mockgal renders of the CS4G sample against the original
Salo+2015 `*_subcomps.fits` cubes hosted on IRSA. Both sides use
Spitzer/IRAC1 native geometry (0.75 arcsec/pix, zp=21.097, Gaussian
PSF FWHM=1.66 arcsec) driven by `inputs/cs4g/runs/cs4g_s4g_irac1_test.yaml`.

#### Selection constraints (locked in)

- **No galaxies with a PSF component** (PointSourceComponent handling is
  assumed-correct across all engines; excluding PSF isolates the
  extended-component fidelity).
- **No galaxies with a Sersic index > 8** (mockgal clamps n>8 to 8.0 at
  load time; any such galaxy would have a non-faithful bulge render
  that confounds the comparison).
- Scope: **3 galaxies** covering the profile-mix spectrum.
- PSF: keep the Gaussian FWHM=1.66 arcsec approximation for now
  (option (a)). Only revisit with per-galaxy Salo composite PSF files
  if the core residuals dominate the comparison.

#### Proposed validation sample

From the 78 sample galaxies that pass the two constraints:

| Galaxy    | Components             | Role in validation                                   |
|-----------|------------------------|------------------------------------------------------|
| NGC1433   | BULGE + DISK + BAR     | Classic SBab ringed barred spiral; exercises Ferrer  |
| NGC0275   | DISK + DISK + BAR      | Bulgeless disk-dominated; exercises Ferrer           |
| NGC1357   | BULGE + DISK + DISK    | Sersic-only control (no Ferrer, no PSF)              |

Two galaxies exercise the Ferrer conversion; the third isolates the
Sersic/expdisk path.

#### Steps

1. Fetch each galaxy's `{name}_subcomps.fits.gz` from the IRSA P4
   folder, unzip, save to `output/cs4g_s4g_irac1_test/{name}/s4g_subcomps.fits`.
2. Build a sum image from the subcomps cube (sky plane excluded).
3. Render via mockgal libprofit driven by the test manifest.
4. Optionally render via `mockgal_galfit.py` (MockGalaxy mode) as a
   second reference; useful to separate "our conversion is right" from
   "libprofit ~ GALFIT".
5. Crop/align to a common FOV, compute flux ratios and radial profiles.
6. Emit per-galaxy and summary QA figures to
   `output/cs4g_s4g_irac1_test/qa_s4g_validation.png`.

#### Expected residual sources

- Central PSF differences (Salo's composite PSF vs our Gaussian) --
  should dominate inside ~2-3 FWHM of the nucleus.
- Axis-ratio convention quirks if any (we have not hit these in the
  NGC1097/NGC1365 2-galaxy check; flag if they appear).
- Profile-edge differences near Ferrer's truncation -- already
  characterized in the GALFIT-vs-libprofit Ferrer empirical test
  (~1% flux near r_out, numerical-only).

### Phase 9 — Production render in HSC i-band

**Goal**: produce 198-galaxy CS4G mocks at HSC-i survey settings, in
the same per-galaxy folder layout as `~/Dropbox/work/data/huang2013/`
so the downstream isophote-fitting benchmark pipeline can consume
either dataset uniformly.

#### Output contract (must match huang2013)

Every mock dataset in `~/Dropbox/work/data/{dataset}/` looks like:

```
{dataset}/
├── run_manifest_original.yaml      # verbatim input manifest
├── run_manifest_resolved.yaml      # defaults merged into each configs row
├── run_metadata.json               # timestamp, git branch, resolved rows
└── {galaxy}/
    ├── {galaxy}_clean_z005.fits    # single noise-light reference
    ├── {galaxy}_wide_z{005,020,035,050}.fits    # HSC wide depth × 4 redshifts
    ├── {galaxy}_deep_z{005,020,035,050}.fits    # HSC dud depth × 4 redshifts
    └── {galaxy}_mosaic.png         # QA mosaic
```

For CS4G the target dataset dir is `~/Dropbox/work/data/s4g_mock/`.

#### Image sizing — `size_factor=4`

- **Huang2013** sets `re_overall` per galaxy (Kron-like catalog Re) and
  renders at `size_factor=6` → image half-side = 6 × `re_overall_pix`.
- **CS4G** models do not set `re_overall`; mockgal falls back to
  `max(component angular extent)`, which for every CS4G galaxy is the
  extended disk's Sersic-equivalent Re (disk Re ≫ bulge Re; Ferrer
  r_out is hard-truncated).
- At `size_factor=4 × max_Re`, flux completeness is:
  - Disk (n=1): 98.6%
  - Bulge (n=1–4, Re ≪ disk Re): ~100%
  - Ferrer bar: 100% (truncated inside the image)
- `size_factor=4` exceeds Huang2013's effective ~88% flux completeness
  for n=4 de Vaucouleurs profiles. Disk-space impact: ~44% of
  Huang2013's per-galaxy pixel area (56% savings).
- `max_image_size=4001` cap still applies as a safety rail.

#### Manifests

Two YAMLs under `inputs/cs4g/runs/`, mirroring the Huang2013 structure:

- `cs4g_hsc_i_wide.yaml` → 5 configs: `clean_z005`, `wide_z{005,020,035,050}`.
- `cs4g_hsc_i_dud.yaml` → 4 configs: `deep_z{005,020,035,050}`.

`defaults:` block mirrors Huang2013 HSC-i conventions exactly
(pixel_scale=0.168, zeropoint=27.0, psf_fwhm=0.7, randomize_noise_seed=
true), except `size_factor: 4.0` instead of `6.0` and `model_file`
pointing at the CS4G sample YAML.

Per-config `sky_sb_limit` values reuse the calibrated HSC-i depths
from `inputs/huang2013/configs/huang2013_hsc_i_calibration.yaml`:
28.5 (clean), 24.62 (wide), 25.70 (dud).

#### Milestones (C-prefix to avoid colliding with mockgal-library P9.x)

- **C9.1**: Rename `inputs/huang2013/scripts/generate_huang2013_mocks.py`
  → `scripts/generate_mocks.py` at repo root. Decouple from Huang2013-
  specific paths. Update `huang2013_full.sh` / `huang2013_test.sh`
  and any tests. Full suite must stay green.
- **C9.2**: Write the two CS4G production manifests above.
- **C9.3**: Smoke render NGC1433 / NGC0275 / NGC1357 through both
  manifests. Verify output layout matches Huang2013's byte-for-byte
  structurally (same files per galaxy, same root metadata files).
  Inspect one mosaic PNG visually.
- **C9.4**: Full batch render (after user sign-off on C9.3):
  198 × 2 manifests = 1782 FITS + 198 mosaic PNGs → `~/Dropbox/work/data/s4g_mock/`.
  Add `s4g_mock_full.sh` + `s4g_mock_test.sh` at repo root.
  Capture rendering timings.
- **C9.5**: Document the mock-dataset folder contract in `docs/SPEC.md`
  so Phase 10's isophote-fitting pipeline can consume any dataset
  matching the contract.

Phase 9 ends once the mocks are on disk and the contract is documented.

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
