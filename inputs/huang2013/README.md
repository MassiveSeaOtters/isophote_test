# Huang2013 Workflow

This directory contains the canonical Huang2013 assets and the production-oriented workflow entry points.

## Layout

- `catalog/`: source Huang et al. (2013) table used to regenerate the model set
- `models/`: canonical 93-galaxy MockGal model file
- `configs/huang2013_test_config.yaml`: libprofit-backed validation config for targeted Huang2013 checks
- `configs/huang2013_hsc_i_calibration.yaml`: optional HSC `i`-band calibration references for Huang2013 noise experiments
- `runs/`: canonical manifest-driven production run definitions
- `scripts/convert_huang2013.py`: catalog-to-model conversion helper
- `scripts/generate_huang2013_mocks.py`: systematic production-oriented mock generator

## Calibration Scope

- `configs/huang2013_hsc_i_calibration.yaml` is a Huang2013 workflow reference file, not a global MockGal default.
- The stored `wide` and `dud` values come from `output/hsc_sky_calibration_full_run/combined_summary.csv`.
- These profiles are only calibrated for HSC `i`-band `coadd/bg` cutouts.
- Production manifests must copy those numeric values inline rather than reference profile names at runtime.

## Manifest Field Notes

- `size_factor`: image half-size in units of `re_overall` after projection to pixels. The final computed image width is `2 * int(size_factor * overall_re_pix) + 1`.
- `size_factor` must be declared on each manifest row. It is intentionally not inherited from manifest-level defaults, so high-z or low-S/N rows can enlarge or shrink their images independently.
- `size_pixels`: optional fixed image size. When present, it overrides `size_factor`.
- `max_image_size`: hard ceiling on the final image dimension. MockGal still computes the natural size from `re_overall` and `size_factor`, or from `size_pixels`, then caps it to this value to avoid oversized outputs.
- `re_overall`: galaxy-level effective-radius anchor in kpc stored in `models/huang2013_models.yaml`. It is currently a flux-weighted average of the component `r_eff_kpc` values and is used as the canonical Huang2013 image-sizing anchor.
- `sky_sb_limit`: 5-sigma surface-brightness depth used for background-limited noise generation. Larger values are fainter and therefore lower-noise.
- `noise_enabled`: should stay `true` for science-like production rows. Fully noise-free rows are discouraged because downstream fitting and image-analysis tools generally assume some nonzero background fluctuation.
- `engine`: `auto` lets MockGal choose the available backend, while `libprofit` forces the libprofit path and therefore depends on `LIBPROFIT_PATH` and `profit-cli` availability.

## Baseline Run Intent

- The baseline production manifest intentionally uses a low-noise `z005_low_noise` row with `sky_sb_limit = 29.0` instead of a perfectly noise-free image.
- The baseline production manifest now sets `size_factor = 6.0` on each row, with `re_overall` as the size anchor instead of the largest component radius.
- This keeps a near-ideal reference image while avoiding the unrealistic zero-noise regime that tends to confuse image-analysis and galaxy-modeling workflows.

## Recommended Output Layout

- Validation smoke runs: `output/huang2013_validation/<run_name>`
- Production-oriented systematic runs: `output/huang2013_production/<run_name>`

## Commands

Small libprofit-backed validation run:

```bash
python mockgal.py \
    --models inputs/huang2013/models/huang2013_models.yaml \
    --config inputs/huang2013/configs/huang2013_test_config.yaml \
    --galaxy "IC 1459" "NGC 1399" \
    --workers 1 \
    -o output/huang2013_validation/libprofit_smoke \
    -v
```

Baseline production smoke run:

```bash
python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_production_baseline.yaml \
    --output output/huang2013_production/huang2013_production_baseline \
    --test
```

Production smoke run using the HSC `i`-band `wide` manifest:

```bash
python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_hsc_i_wide.yaml \
    --output output/huang2013_production/huang2013_hsc_i_wide \
    --test \
    --config-name z005_clean z020_hsc_i_wide
```

Production run for a selected subset with the HSC `i`-band `dud` manifest:

```bash
python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_hsc_i_dud.yaml \
    --output output/huang2013_production/huang2013_hsc_i_dud_subset \
    --galaxies "IC 1459" "NGC 3923" "NGC 4472" \
    --config-name z005_clean z050_hsc_i_dud
```

Real baseline production run for the first Huang2013 galaxy (`ESO 185-G054`):

```bash
uv run python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_production_baseline.yaml \
    --output ~/Dropbox/work/data/huang2013/huang2013_production_baseline_first_galaxy \
    --galaxies "ESO 185-G054"
```

Real baseline production run for the full Huang2013 sample:

```bash
uv run python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_production_baseline.yaml \
    --output ~/Dropbox/work/data/huang2013/huang2013_production_baseline
```

## Verification Note

- Export `LIBPROFIT_PATH=/Users/shuang/Dropbox/work/project/otters/isophote_test/libprofit/mbp` manually in this shell before libprofit-backed validation commands.
- Re-verify `LIBPROFIT_PATH` in the current shell before relying on the libprofit-backed validation command.
- The production generator writes `run_manifest_original.yaml`, `run_manifest_resolved.yaml`, and `run_metadata.json` into the output directory for reproducibility.
