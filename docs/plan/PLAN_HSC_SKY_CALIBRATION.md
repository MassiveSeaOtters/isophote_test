# HSC Sky Calibration Prototype Plan

## Goal

Define and validate a small workflow that uses real HSC sky-object positions to estimate background statistics for MockGal noise calibration.

## Inputs

- `inputs/data/cosmos_dud_sky.csv`
- `inputs/data/cosmos_wide_sky.csv`
- `/Users/shuang/Dropbox/work/project/otters/hsc_sandbox/side1`

## Current Findings

- The two COSMOS catalogs are independent sky-region samples for the HSC `wide` and `dud` layers.
- `side1` already has live-validated cutout support for:
  - `s21a_dud2`
  - `s23b_wide`
- `side1` batch inputs only need `id,ra,dec`; rerun, filter, and cutout size can be supplied as CLI defaults.
- This workflow should request only the image and variance planes. Mask and PSF products are not required.

## Prototype Scope

1. Randomly select sky positions from each catalog independently, with a default sample size of 30 per layer.
2. Download small `i`-band `coadd/bg` cutouts for:
   - `s21a_dud2`
   - `s23b_wide`
3. Use `5 x 5 arcsec` boxes:
   - `half_size_width = 2.5arcsec`
   - `half_size_height = 2.5arcsec`
4. Measure:
   - sigma-clipped local median per cutout
   - median-centered image-plane pixel distributions
   - pooled RMS across all centered pixels in a layer
   - variance-plane median and implied sigma
5. Convert measured values into MockGal-compatible estimates:
   - primary: `sky_sb_limit` from pooled centered RMS
   - secondary: exploratory `sky_sb_value` summary from original cutout medians

## Validation Strategy

- Start with a small validation run before scaling to 30 regions per layer.
- Keep cutouts small, with default half-size `2.5arcsec`.
- Treat `coadd/bg` as the primary download product for this workflow.
- Center each cutout locally before pooling pixels across a layer.
- Do not run broad catalog downloads until the one-object path is verified locally.

## Open Risks

- `inputs/data/` is still untracked.
- The shell environment still points `LIBPROFIT_PATH` to `/Users/mac/...`, while local validation required `/Users/shuang/...`.
- HSC coadds are stacked products, so any inferred `gain` is an effective calibration parameter rather than a detector-native quantity.
