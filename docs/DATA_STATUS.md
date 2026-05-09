# Available Data And Test Status

This note records the current mock-data and isophote-test artifacts available
for this repository as of 2026-05-09. It is intended as a starting point for
future development sessions and for users who need to know what data exist,
where they live, and what has already been validated.

## Repository Role

This repository stores the mock-galaxy generation code, model definitions,
run manifests, conversion scripts, QA scripts, and documentation. The large
rendered FITS datasets and downstream isophote-analysis campaign products are
not primarily stored in the repository.

The canonical generated dataset contract is defined in `docs/SPEC.md`.
Each production mock dataset has root-level manifest metadata and one
subdirectory per rendered galaxy:

```text
{dataset_root}/
|-- run_manifest_original.yaml
|-- run_manifest_resolved.yaml
|-- run_metadata.json
`-- {galaxy}/
    |-- {galaxy}_{config}.fits
    `-- {galaxy}_mosaic.png
```

## Production Mock Datasets

### Huang2013

Purpose: multi-component models of local early-type galaxies from Huang et al.
(2013), used as the main systematic early-type mock dataset.

Repository assets:

- Source catalog: `inputs/huang2013/catalog/huang2013_cgs_model.txt`
- Canonical model YAML: `inputs/huang2013/models/huang2013_models.yaml`
- Run manifests: `inputs/huang2013/runs/`
- Documentation: `inputs/huang2013/README.md`

Current model contents:

- 93 galaxies
- 292 Sersic components
- 93 `re_overall` image-sizing anchors
- 29 flagged galaxies

Current production data roots found on disk:

- `/Volumes/galaxy/isophote/huang2013`
- `/Users/shuang/Dropbox/work/data/huang2013`

Current production render inventory:

- 93 galaxy directories
- 837 FITS files
- 93 mosaic PNG files
- Root provenance files: `run_manifest_original.yaml`,
  `run_manifest_resolved.yaml`, `run_metadata.json`

The production grid has 9 configurations per galaxy:

- `clean_z005`
- `wide_z005`, `wide_z020`, `wide_z035`, `wide_z050`
- `deep_z005`, `deep_z020`, `deep_z035`, `deep_z050`

The latest root metadata observed for both Huang2013 locations records the
last manifest run as `inputs/huang2013/runs/huang2013_hsc_i_dud.yaml`, with
93 galaxies and 4 DUD/deep rows. This is expected because multiple manifests
are run into the same dataset root and root metadata are overwritten by the
last manifest run.

### S4G Mock / CS4G

Purpose: multi-component models of nearby galaxies, mostly late-type disks,
derived from Salo et al. S4G P4 GALFIT outputs and CS4G catalog metadata.

Repository assets:

- CS4G catalog products: `inputs/cs4g/cs4g_catalog.fits`,
  `inputs/cs4g/cs4g_catalog.tbl`, `inputs/cs4g/cs4g_2025.fit`
- Candidate table: `inputs/cs4g/cs4g_candidates.csv`
- P4 index: `inputs/cs4g/cs4g_p4_index.csv`
- Local P4 `.outgal` mirror: `inputs/cs4g/p4/`
- Parsed models: `inputs/cs4g/cs4g_models.json`
- Converted mockgal components: `inputs/cs4g/cs4g_components.json`
- Final sample table: `inputs/cs4g/cs4g_sample.csv`
- Canonical model YAML: `inputs/cs4g/models/cs4g_sample_models.yaml`
- Run manifests: `inputs/cs4g/runs/`
- Planning and rationale: `docs/plan/PLAN_CS4G_SAMPLE.md`

Current model contents:

- 300 size-aware sample galaxies
- 523 Sersic components
- 260 Ferrer components
- 130 PSF components
- Metadata band: predicted SDSS `i`-band AB
- Sample selection: size-aware at `z = 0.1`, requiring production image size
  larger than 75 pixels and preferring complex models

Intermediate CS4G inventory:

- 1642 candidate galaxies plus CSV header
- 1208 locally mirrored `.outgal` files
- 1198 parsed and kept model records
- 10 edgedisk-skipped records
- 300 final sampled galaxies

Current production data roots found on disk:

- Current target: `/Volumes/galaxy/isophote/s4g_mock`
- Older 198-galaxy mirror: `/Users/shuang/Dropbox/work/data/s4g_mock`

Current production render inventory at `/Volumes/galaxy/isophote/s4g_mock`:

- 300 galaxy directories
- 1500 FITS files
- 300 mosaic PNG files
- Root provenance files: `run_manifest_original.yaml`,
  `run_manifest_resolved.yaml`, `run_metadata.json`

The current production grid has 5 configurations per galaxy:

- `clean_z005`
- `wide_z005`, `wide_z010`
- `deep_z005`, `deep_z010`

The older Dropbox mirror has 198 galaxies and the older 9-configuration grid
matching the earlier Huang2013-style redshift set. Prefer
`/Volumes/galaxy/isophote/s4g_mock` for current work.

## Noise And Survey Calibration

HSC `i`-band sky calibration artifacts are stored under:

- `output/hsc_sky_calibration_full_run/`

Important files:

- `combined_summary.csv`
- `wide/layer_summary_wide.csv`
- `dud/layer_summary_dud.csv`
- QA figures for real cutouts, simulated noise, side-by-side comparisons,
  and pixel distributions

Measured depth values copied into production manifests:

- HSC wide: `sky_sb_limit = 24.62395438263279`
- HSC DUD/deep: `sky_sb_limit = 25.697653749397546`
- Clean reference rows use `sky_sb_limit = 28.5`

These values are scoped to HSC `i`-band mock-generation workflows and should
stay explicit in run manifests.

## Mock-Generation QA

### Huang2013 Smoke And Validation Outputs

Small Huang2013 validation outputs exist under:

- `output/huang2013/`
- `output/huang2013_doc_validation/`
- `output/huang2013_production/`

These include early smoke renders, documentation-validation FITS files, and
manifest smoke-run metadata. The production-quality full dataset is outside
the repo under the data roots listed above.

### S4G Native IRAC1 Validation

The main CS4G/S4G render-validation artifact is:

- `output/cs4g_s4g_irac1_test/`

It compares mockgal renders against original Salo+2015 `_subcomps.fits`
reference cubes for three representative galaxies:

- `NGC1433`: Sersic + Sersic + Ferrer
- `NGC0275`: Sersic + Ferrer + Ferrer
- `NGC1357`: Sersic + Sersic + Sersic

Available files:

- Per-galaxy `mockgal.fits`
- Per-galaxy `s4g_reference.fits`
- Per-galaxy `s4g_subcomps.fits`
- Per-galaxy `s4g_psf_composite.fits`
- `qa_summary.json`
- `qa_s4g_validation.png`

Final validation metrics after the color-shift fix:

| Galaxy | flux_ratio | corr | peak_ratio | Main interpretation |
|---|---:|---:|---:|---|
| `NGC1433` | 0.996 | 0.974 | 1.96 | Good total-flux agreement; central PSF/core residuals remain |
| `NGC0275` | 0.991 | 0.9996 | 1.01 | Excellent agreement for bulgeless Ferrer-bearing case |
| `NGC1357` | 0.989 | 0.877 | 3.98 | Flux agreement is good; peak residual dominated by Gaussian vs composite PSF |

The key lesson is that `inputs/cs4g/models/cs4g_sample_models.yaml` stores
predicted `i`-band absolute magnitudes. Validation against IRAC1 reference
cubes must add back `color_shift_3p6_minus_i` before rendering.

### Libprofit Versus GALFIT QA

Additional CS4G renderer QA exists under:

- `output/cs4g_qa/`

It compares libprofit and GALFIT reference renders for `NGC1097` and
`NGC1365`. The observed flux ratios are close to unity:

- `NGC1097`: flux ratio 0.99898, correlation 0.9765
- `NGC1365`: flux ratio 0.99557, correlation 0.9207

## Isophote-Analysis Campaign Artifacts

Downstream isophote-analysis campaigns have been run outside the repo on both
production datasets. These campaigns compare:

- `isoster`
- `photutils`
- `autoprof`

Each campaign stores per-galaxy outputs, per-tool cross-arm summaries,
default-arm cross-tool tables, and aggregate cross-scenario analyses.

### Huang2013 Campaigns

Location:

- `/Volumes/galaxy/isophote/huang2013/_campaigns`

Campaign scenarios:

- `huang2013_clean_z005`
- `huang2013_wide_z005`, `huang2013_wide_z020`,
  `huang2013_wide_z035`, `huang2013_wide_z050`
- `huang2013_deep_z005`, `huang2013_deep_z020`,
  `huang2013_deep_z035`, `huang2013_deep_z050`

Each scenario has 279 default cross-tool rows:

- 93 galaxies x 3 tools

Representative per-galaxy artifact structure:

```text
{campaign}/{dataset}/{galaxy}__{config}/
|-- MANIFEST.json
|-- isoster/
|   |-- inventory.fits
|   |-- cross_arm_table.csv
|   |-- cross_arm_table.md
|   `-- cross_arm_overlay.png
|-- photutils/
|-- autoprof/
`-- cross/
    `-- cross_tool_comparison.png
```

Aggregate analysis files include:

- `_analysis/cross_tool_composite/cross_tool_pooled_ranking.md`
- `_analysis/cross_tool_composite/cross_tool_best_available.md`
- `_analysis/cross_tool_composite/cross_tool_heatmap.pdf`
- `_analysis/cross_scenario_audit_isoster/`
- `_analysis/cross_scenario_audit_photutils/`
- `_analysis/cross_scenario_audit_autoprof/`
- `_analysis/cross_tool_extended_metrics/`

Observed aggregate headline from `cross_tool_best_available.md`:

- `photutils/aggressive_clip` wins `clean_z005`.
- `autoprof/deep` wins low-redshift wide/deep scenarios.
- `isoster` regularized or stacked arms win most higher-redshift wide/deep
  scenarios.

### S4G Campaigns

Location:

- `/Volumes/galaxy/isophote/s4g_mock/_campaigns`

Campaign scenarios:

- `s4g_clean_z005`
- `s4g_wide_z005`, `s4g_wide_z010`
- `s4g_deep_z005`, `s4g_deep_z010`

Most scenario tables have 900 default cross-tool rows:

- 300 galaxies x 3 tools

One observed exception:

- `s4g_wide_z005` has 899 default cross-tool rows, so one default tool result
  appears missing or filtered and should be checked before any strict
  completeness claim.

Aggregate analysis files mirror the Huang2013 campaign structure.

Observed aggregate headline from `cross_tool_best_available.md`:

- `photutils/aggressive_clip` wins `clean_z005`, `deep_z005`, and narrowly
  `deep_z010`.
- `autoprof/deep` is tied or near-tied in `wide_z005`.
- `isoster/reg_outer_center_heavy_5x` wins or ties in `wide_z010`.

Pooled ranking highlights:

- S4G `isoster` top arms reach clean fractions near 0.958 across the
  1500-row pooled grid.
- S4G `photutils/aggressive_clip` reaches a clean fraction near 0.970 on its
  scored pooled rows.
- S4G `autoprof/deep` reaches a clean fraction near 0.907.

## In-Repo Test Coverage

The repository includes focused unit and integration tests for the generator,
component models, parser utilities, and calibration helpers:

- `tests/test_mockgal.py`
- `tests/test_ferrer_component.py`
- `tests/test_pointsource_component.py`
- `tests/test_mockgal_galfit.py`
- `tests/test_generate_mocks.py`
- `tests/test_cs4g_downsample.py`
- `tests/test_hsc_sky_calibration.py`
- `tests/test_galfit_io.py`
- `tests/test_galfit_constraints.py`
- `tests/test_galfit_integration.py`

At inspection time, 217 test functions were present.

## Practical Notes For Future Work

- Prefer `/Volumes/galaxy/isophote/huang2013` and
  `/Volumes/galaxy/isophote/s4g_mock` for current rendered data.
- Treat `/Users/shuang/Dropbox/work/data/s4g_mock` as an older 198-galaxy
  S4G production mirror.
- Use `scripts/generate_mocks.py` and the checked-in run manifests to
  regenerate production datasets.
- Do not infer complete run provenance from root `run_metadata.json` alone
  after multi-manifest production; it records the last manifest run only.
- For S4G native IRAC1 validation, remember the band convention:
  the production YAML stores predicted `i`-band magnitudes, while S4G P4
  reference cubes are IRAC1.
- For high-fidelity S4G core validation, use the per-galaxy composite PSF
  files that were fetched during QA instead of relying on the simple Gaussian
  PSF approximation.
- For full campaign interpretation, start with the aggregate Markdown reports
  under each dataset's `_campaigns/_analysis/` directory, then inspect
  per-galaxy cross-tool comparison PNGs for representative failures.
