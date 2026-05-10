# Plan: v1.1 QA And Evaluation Refresh For Mock Campaigns

Date: 2026-05-09

## Goal

Refresh the existing Huang2013 and S4G mock isophote-analysis campaigns so
their QA figures, per-arm model evaluations, flags, and cross-tool rankings
match the current `isoster` benchmark standard.

The primary campaign products are outside this repository:

- `/Volumes/galaxy/isophote/huang2013/_campaigns`
- `/Volumes/galaxy/isophote/s4g_mock/_campaigns`

The implementation belongs mostly in the sibling `isoster` repository because
the exhausted-benchmark runners, QA plotting functions, inventory schema, and
model-evaluation logic are maintained there.

## Current State

The existing campaign trees contain enough persisted artifacts to refresh
metrics and QA without immediately refitting every arm:

- `MANIFEST.json`
- `profile.fits`
- `model.fits`
- `run_record.json`
- `inventory.fits`
- `cross_arm_overlay.png`
- `cross_tool_comparison.png`

Sample campaign inventories are still on the older 44-column schema. Sample
`run_record.json` files contain old residual metrics and lowercase flags, and
are missing the v1.1 fields such as `F_ref`, `R_ref_used_pix`,
`r_inner_floor_pix`, sigma-normalized residuals, azimuthal metrics, portable
uppercase flags, `cross_tool_score`, and `cross_tool_score_simple`.

## Standard To Match

The refresh follows the updated `isoster` documentation:

- `docs/06-qa-functions.md`
- `docs/09-exhausted-benchmark.md`
- `docs/01-user-guide.md`
- `docs/agent/qa-figures.md`

Key requirements:

- QA figures must visualize the same profile/model/residual products used for
  metrics and ranking.
- Residual metrics must use the v1.1 radial zones anchored to
  `R_ref_used_pix`.
- Cross-tool ranking must use `cross_tool_score` / `cross_tool_score_simple`,
  not method-specific diagnostics.
- Per-arm ranking within a tool can use `composite_score`.
- Flags should use the current portable uppercase flag names and severities.
- QA plots should use calibrated SB when zeropoint and pixel scale are
  available, `SMA**0.25` as the standard radial coordinate, consistent
  clipping, and explicit stop-code/quality markers.

## Special Note: Photutils Harmonics

Some previous photutils runs may have rendered 2-D models with harmonics
disabled. For model-image refresh, photutils arms should be audited for the
stored harmonic setting. If harmonics were disabled in the past run, the
refresh should support rebuilding the photutils `model.fits` with harmonics
enabled before recomputing residual metrics and QA. This avoids ranking a
new-standard profile against an outdated low-order model image.

## Implementation Steps

1. Add an artifact-level refresh command in `isoster`.
   - Input: one campaign root and dataset name.
   - Optional filters: scenario, galaxy directory, tool, arm, dry-run, limit,
     and write/no-write.
   - Read existing `MANIFEST.json`, source FITS image, `profile.fits`,
     `model.fits`, and `run_record.json`.
   - Recompute v1.1 model metrics from persisted image/model/profile products.
   - Recompute current quality flags.
   - Rewrite `run_record.json` and `inventory.fits` only when `--write` is
     supplied.

2. Add photutils model refresh support.
   - Detect old photutils model settings from `config.yaml` or `run_record.json`.
   - If harmonics were disabled and a refresh flag is supplied, rebuild the
     photutils `model.fits` with harmonics enabled using the stored profile.
   - Record whether a model image was reused or rebuilt.

3. Regenerate scores and tables.
   - Apply current `composite_score` within each tool.
   - Rebuild per-tool `inventory.fits` and cross-arm tables.
   - Rebuild per-galaxy cross-tool tables with `cross_tool_score` and
     `cross_tool_score_simple`.

4. Regenerate QA figures.
   - Reuse or extend `benchmarks/exhausted/campaigns/rerender_qa.py`.
   - Ensure the regenerated PNGs use the refreshed metrics and model images.
   - Pilot `asinh` SB scaling on noisy/low-S/N scenarios before making it a
     broad default.

5. Rebuild aggregate analyses.
   - Re-run cross-scenario audits for `isoster`, `photutils`, and `autoprof`.
   - Re-run cross-tool composite and extended-metric summaries.
   - Investigate the known `s4g_wide_z005` 899-row cross-tool-table exception.

6. Validate in stages.
   - First use a smoke subset: one Huang2013 clean case, one Huang2013 noisy
     case, `NGC1433__clean_z005`, and one S4G wide/deep case.
   - Confirm the new inventory schema, v1.1 metric fields, uppercase flags,
     regenerated QA PNGs, and cross-tool scores.
   - Only then run campaign-scale refresh.

## Initial Success Criteria

- A dry-run command can enumerate existing campaign arms without writing.
- A write-mode smoke run refreshes one galaxy/tool/arm and produces v1.1
  metrics in `run_record.json`.
- The refreshed inventory uses the current `isoster` schema.
- QA figures regenerate from the same model files used for residual metrics.

## Implementation Progress

### 2026-05-09

Initial implementation was started on branch `qa-evaluation-refresh`.

In the sibling `isoster` repository:

- Added `benchmarks/exhausted/campaigns/refresh_model_evaluation.py`.
- Added focused tests in `tests/unit/test_refresh_model_evaluation.py`.
- The new command can enumerate campaign galaxy directories in dry-run mode.
- The command can refresh v1.1 metrics and flags from existing `profile.fits`,
  `model.fits`, `MANIFEST.json`, and `run_record.json`.
- It rewrites `run_record.json`, per-tool `inventory.fits`,
  `cross_arm_table.csv/md`, and scenario-level `cross_tool_table.csv/md` only
  when `--write` is supplied.
- It supports `--refresh-photutils-harmonic-models`, which rebuilds photutils
  `model.fits` from stored `profile.fits` with harmonics enabled before
  recomputing residual metrics.

Validation completed:

- `uv run pytest tests/unit/test_refresh_model_evaluation.py`
- `uv run pytest tests/unit/test_refresh_model_evaluation.py tests/unit/test_exhausted_analysis_v11.py`
- `uv run ruff check benchmarks/exhausted/campaigns/refresh_model_evaluation.py tests/unit/test_refresh_model_evaluation.py`
- Dry-run on real S4G campaign path for `NGC1433__clean_z005`.
- Write-mode refresh on a temporary copy of `NGC1433__clean_z005` for
  `isoster/ref_default` and `photutils/baseline_median`; no production
  campaign files were modified.

## Proposed Next Step

Review the temporary-copy smoke output, then extend the command to rerender QA
figures after metric/model refresh so the visual products and updated residual
metrics are regenerated in the same workflow.
