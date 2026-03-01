---
date: 2026-03-01
repo: isophote_test
branch: main
tags:
  - journal
  - huang2013
  - hsc
  - calibration
  - docs
---

## Progress

- Merged `feature/hsc-sky-cutout-prototype` into `main` as `b0bcc12 Add HSC sky calibration workflow and validation assets`.
- Added `inputs/data/cosmos_wide_sky.csv` and `inputs/data/cosmos_dud_sky.csv` to the repository as the canonical HSC sky-region catalogs for this calibration work.
- Added `scripts/hsc_sky_calibration.py` to sample sky regions, batch-download HSC cutouts through `side1`, sigma-clip each cutout, recenter local backgrounds, pool pixels by layer, and derive MockGal-facing calibration summaries.
- Added `tests/test_hsc_sky_calibration.py` to cover catalog loading, reproducible sampling, batch downloader argument construction, cutout measurement, and summary-value conversion.
- Added `docs/plan/PLAN_HSC_SKY_CALIBRATION.md` and updated `docs/todo.md` plus `docs/LESSON.md` to capture the workflow and conclusions.
- Live-validated the HSC downloader path with:
  - `python scripts/hsc_sky_calibration.py --layer wide --sample-count 3 --output-dir output/hsc_sky_calibration_validation --overwrite`
  - `python scripts/hsc_sky_calibration.py --layer both --sample-count 3 --no-qa --output-dir output/hsc_sky_calibration_validation_both --overwrite`
  - `python scripts/hsc_sky_calibration.py --layer both --sample-count 30 --output-dir output/hsc_sky_calibration_full_run --overwrite`
- Produced final full-run calibration outputs and QA products under `output/hsc_sky_calibration_full_run/` for both `wide` and `dud`.

## Lessons Learned

- The intended HSC sky workflow should use the `coadd/bg` cutout product with image and variance planes only; mask and PSF retrieval are unnecessary for this blank-sky calibration task.
- For this use case, each cutout should be sigma-clipped and centered on its local median before pixels are pooled across regions.
- The pooled centered RMS provides a stable mapping into MockGal `sky_sb_limit`, while per-cutout background medians are only suitable as secondary exploratory summaries.
- The full 30-region run gave layer-specific `i`-band calibrations of about `24.62` for `wide` and `25.70` for `dud`, but these should be documented as test results rather than hardwired as general defaults.

## Key Issues

- The current calibration result is only for the HSC `i` band and should be referenced in Huang2013 test documentation instead of being promoted to a global default.
- In this agent shell, `LIBPROFIT_PATH` still resolves to `/Users/mac/.../libprofit/mbp`; that did not affect the HSC calibration run, but it should be rechecked before later MockGal/libprofit validation.
- The next session should reorganize the Huang2013 test structure and prepare the repo for a cleaner production-run workflow.
