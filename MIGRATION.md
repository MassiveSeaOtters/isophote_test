# Migration Guide

This repository now uses canonical subdirectories under `inputs/` instead of a flat layout.

## Path Migration

Preferred canonical paths:

- `inputs/huang2013/models/huang2013_models.yaml`
- `inputs/huang2013/configs/huang2013_test_config.yaml`
- `inputs/huang2013/catalog/huang2013_cgs_model.txt`
- `inputs/huang2013/scripts/convert_huang2013.py`
- `scripts/generate_mocks.py` (dataset-agnostic; was `inputs/huang2013/scripts/generate_huang2013_mocks.py` before the Phase 9 rename+lift)
- `inputs/examples/models/example_models.yaml`
- `inputs/examples/configs/example_image_config.yaml`
- `inputs/demos/*.py`

The old flat `inputs/` paths have been removed. Update docs, scripts, and automation to the canonical paths under `inputs/huang2013/`, `inputs/examples/`, and `inputs/demos/`.

## Filename Convention

Use `sanitize_filename()` for outputs:

- Correct: `NGC3923_clean.fits`
- Incorrect: `NGC_3923_clean.fits`

Galaxy-name spaces are removed, and the only underscore should separate the sanitized galaxy name from the config name.

## Huang2013 Workflow Updates

The canonical systematic Huang2013 generator now assumes:

- dynamic image sizing with `size_factor = 16`
- `sky_sb_limit = 24.5` for noisy mocks
- a 4001-pixel safety cap
- one QA mosaic PNG per galaxy

## Recommended Command Updates

Before:

```bash
python mockgal.py \
    --models inputs/huang2013_models.yaml \
    --config inputs/huang2013_test_config.yaml \
    -o output/
```

After:

```bash
python mockgal.py \
    --models inputs/huang2013/models/huang2013_models.yaml \
    --config inputs/huang2013/configs/huang2013_test_config.yaml \
    -o output/
```
