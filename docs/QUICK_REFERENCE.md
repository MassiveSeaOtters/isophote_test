# MockGal Quick Reference

## Canonical Paths

- Huang2013 models: `inputs/huang2013/models/huang2013_models.yaml`
- Huang2013 test config: `inputs/huang2013/configs/huang2013_test_config.yaml`
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
    --output output/huang2013_systematic \
    --test
```

## Critical Conventions

- Use `sanitize_filename()` for output names.
- Huang2013 `VMag` values are already absolute magnitudes.
- Prefer `--workers 1` for Huang2013 validation on constrained machines.
- The systematic Huang2013 generator currently uses `size_factor = 16`, `sky_sb_limit = 24.5`, and a 4001-pixel cap.

## Where To Look

- Project map: `CLAUDE.md`
- Agent workflow rules: `AGENTS.md`
- Inputs inventory: `inputs/README.md`
- Local `libprofit` build notes: `docs/LIBPROFIT_COMPILE.md`
