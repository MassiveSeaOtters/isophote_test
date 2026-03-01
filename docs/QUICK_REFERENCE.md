# MockGal Quick Reference

## Canonical Paths

- Huang2013 models: `inputs/huang2013/models/huang2013_models.yaml`
- Huang2013 test config: `inputs/huang2013/configs/huang2013_test_config.yaml`
- Huang2013 HSC `i`-band references: `inputs/huang2013/configs/huang2013_hsc_i_calibration.yaml`
- Huang2013 production manifests: `inputs/huang2013/runs/`
- Huang2013 converter: `inputs/huang2013/scripts/convert_huang2013.py`
- Huang2013 systematic mocks: `inputs/huang2013/scripts/generate_huang2013_mocks.py`
- Lessons learned: `docs/LESSON.md`

## Essential Commands

Run tests:

```bash
pytest tests/test_mockgal.py -v
```

Run benchmarks:

```bash
python benchmarks/bench_engines.py
```

Small Huang2013 validation run:

```bash
export LIBPROFIT_PATH=/Users/shuang/Dropbox/work/project/otters/isophote_test/libprofit/mbp
python mockgal.py \
    --models inputs/huang2013/models/huang2013_models.yaml \
    --config inputs/huang2013/configs/huang2013_test_config.yaml \
    --galaxy "ESO 185-G054" "ESO 221-G026" "IC 1459" "IC 1633" "IC 2006" \
    --workers 1 \
    -o output/huang2013_test \
    -v
```

Systematic Huang2013 mocks:

```bash
python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_production_baseline.yaml \
    --output output/huang2013_production/huang2013_production_baseline \
    --test
```

Systematic Huang2013 mocks with the HSC `i`-band `wide` manifest:

```bash
python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_hsc_i_wide.yaml \
    --output output/huang2013_production/huang2013_hsc_i_wide \
    --test \
    --config-name z005_clean z020_hsc_i_wide
```

## Critical Conventions

- Use `sanitize_filename()` for output names.
- Huang2013 `VMag` values are already absolute magnitudes.
- Prefer `--workers 1` for Huang2013 validation on constrained machines.
- The systematic Huang2013 generator now requires an explicit run manifest under `inputs/huang2013/runs/`.
- `size_factor` sets the half-size relative to Huang2013 `re_overall`; `max_image_size` is the explicit cap on the final image dimension.
- The baseline production manifest uses `size_factor = 6`, `max_image_size = 4001`, `24.5` for its main noisy rows, and `29.0` for its low-noise reference row.
- HSC `i`-band calibration values should be copied inline into dedicated manifests rather than referenced dynamically at runtime.

## Where To Look

- Project map: `CLAUDE.md`
- Agent workflow rules: `AGENTS.md`
- Inputs inventory: `inputs/README.md`
- Local `libprofit` build notes: `docs/LIBPROFIT_COMPILE.md`
