# Huang2013 Run Manifests

This directory contains reproducible production-oriented Huang2013 run manifests.

## Rules

- One manifest file defines one coherent Huang2013 run family.
- Each row under `configs` defines one mock configuration with explicit numeric values.
- Use descriptive config names that explain the scenario rather than opaque `mock1`-style numbering.
- Keep galaxy subset selection in the CLI. Do not embed it in the production manifest.
- Copy HSC-derived values into the manifest rows directly; do not reference calibration profile names at runtime.
- Prefer low-noise reference rows over perfectly noise-free rows for production-like algorithm tests.
- Set `size_factor` on every row explicitly. Do not rely on a manifest-wide `size_factor`, because different rows may need different image extents.

## Included Manifests

- `huang2013_production_baseline.yaml`: baseline production-oriented four-row set using `sky_sb_limit = 24.5`
- `huang2013_production_baseline.yaml` also includes `z005_low_noise` with `sky_sb_limit = 29.0` as the near-ideal reference row
- `huang2013_hsc_i_wide.yaml`: same structure with HSC `i`-band `wide` depth values copied inline
- `huang2013_hsc_i_dud.yaml`: same structure with HSC `i`-band `dud` depth values copied inline

## Example Command

```bash
python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_production_baseline.yaml \
    --output output/huang2013_production/huang2013_production_baseline \
    --test
```
