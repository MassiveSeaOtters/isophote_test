# Inputs Directory

This directory contains the canonical input assets for MockGal.

## Structure

```text
inputs/
├── huang2013/
│   ├── README.md
│   ├── catalog/
│   │   └── huang2013_cgs_model.txt
│   ├── configs/
│   │   ├── huang2013_hsc_i_calibration.yaml
│   │   └── huang2013_test_config.yaml
│   ├── models/
│   │   └── huang2013_models.yaml
│   ├── runs/
│   │   ├── README.md
│   │   ├── huang2013_hsc_i_dud.yaml
│   │   ├── huang2013_hsc_i_wide.yaml
│   │   ├── huang2013_publication_single_band.yaml
│   │   └── huang2013_production_baseline.yaml
│   └── scripts/
│       ├── convert_huang2013.py
│       └── generate_huang2013_mocks.py
├── examples/
│   ├── configs/
│   │   └── example_image_config.yaml
│   └── models/
│       └── example_models.yaml
└── demos/
    ├── api_call_demo.py
    ├── demo_visualization.py
    ├── huang2013_noise_sblimit_demo.py
    └── huang2013_noise_sbvalue_demo.py
```

## Huang2013 Test Suite

Core files:

- `inputs/huang2013/README.md`
- `inputs/huang2013/models/huang2013_models.yaml`
- `inputs/huang2013/configs/huang2013_test_config.yaml`
- `inputs/huang2013/runs/huang2013_production_baseline.yaml`
- `inputs/huang2013/scripts/generate_huang2013_mocks.py`
- `mockgal.py`

Supporting source assets:

- `inputs/huang2013/catalog/huang2013_cgs_model.txt`
- `inputs/huang2013/scripts/convert_huang2013.py`

Typical validation run:

```bash
python mockgal.py \
    --models inputs/huang2013/models/huang2013_models.yaml \
    --config inputs/huang2013/configs/huang2013_test_config.yaml \
    --galaxy "ESO 185-G054" "ESO 221-G026" "IC 1459" "IC 1633" "IC 2006" \
    --workers 1 \
    -o output/huang2013_test \
    -v
```

Systematic mock generation:

```bash
python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_production_baseline.yaml \
    --output output/huang2013_production/huang2013_production_baseline \
    --test
```

Current generator settings:

- HSC-like pixel scale: `0.168` arcsec/pixel
- Gaussian PSF FWHM: `0.7` arcsec
- Dynamic sizing: `size_factor = 6` against Huang2013 `re_overall`
- Image-size cap can now be recorded explicitly as `max_image_size` in run manifests
- Baseline production manifests use `sky_sb_limit = 24.5`
- Maximum image size: `4001`
- One QA mosaic PNG per galaxy
- Optional HSC `i`-band `wide` and `dud` depth references live in `inputs/huang2013/configs/huang2013_hsc_i_calibration.yaml`
- Production runs are now defined by manifest files under `inputs/huang2013/runs/`
- The publication manifest adds genuinely noiseless images and deterministic,
  distinct per-galaxy noise seeds without changing historical manifests.

## Examples And Demos

- `inputs/examples/` contains small reusable YAML files for CLI and API examples.
- `inputs/demos/` contains one-off scripts for API usage, visualization, and targeted Huang2013 noise experiments.
- Demo scripts are not part of the canonical Huang2013 workflow.

## File Inventory

| Path | Type | Role | Required For Huang2013 Test Suite |
|---|---|---|---|
| `inputs/huang2013/README.md` | Markdown doc | Huang2013 workflow layout and commands | No |
| `inputs/huang2013/catalog/huang2013_cgs_model.txt` | ASCII catalog | Source Huang et al. (2013) catalog | No |
| `inputs/huang2013/models/huang2013_models.yaml` | YAML model file | Canonical 93-galaxy Huang2013 model set | Yes |
| `inputs/huang2013/configs/huang2013_test_config.yaml` | YAML config file | Canonical Huang2013 validation config | Yes |
| `inputs/huang2013/configs/huang2013_hsc_i_calibration.yaml` | YAML config file | Optional Huang2013-only HSC `i`-band calibration references | No |
| `inputs/huang2013/runs/README.md` | Markdown doc | Manifest-driven Huang2013 production workflow notes | No |
| `inputs/huang2013/runs/huang2013_production_baseline.yaml` | YAML manifest | Baseline Huang2013 production run set | No |
| `inputs/huang2013/runs/huang2013_hsc_i_wide.yaml` | YAML manifest | HSC `i`-band `wide` Huang2013 production run set | No |
| `inputs/huang2013/runs/huang2013_hsc_i_dud.yaml` | YAML manifest | HSC `i`-band `dud` Huang2013 production run set | No |
| `inputs/huang2013/runs/huang2013_publication_single_band.yaml` | YAML manifest | Publication noiseless/wide/deep single-band run set | No |
| `inputs/huang2013/scripts/convert_huang2013.py` | Python script | Convert source catalog to YAML or JSON | No |
| `inputs/huang2013/scripts/generate_huang2013_mocks.py` | Python script | Generate systematic Huang2013 mocks | No |
| `inputs/examples/models/example_models.yaml` | YAML model file | Small example galaxies | No |
| `inputs/examples/configs/example_image_config.yaml` | YAML config file | Example image settings | No |
| `inputs/demos/api_call_demo.py` | Python demo | Direct API usage example | No |
| `inputs/demos/demo_visualization.py` | Python demo | Visualization example | No |
| `inputs/demos/huang2013_noise_sblimit_demo.py` | Python demo | Gaussian-noise Huang2013 experiment | No |
| `inputs/demos/huang2013_noise_sbvalue_demo.py` | Python demo | Poisson-noise Huang2013 experiment | No |

## Regeneration

Regenerate the Huang2013 YAML models:

```bash
python inputs/huang2013/scripts/convert_huang2013.py \
    inputs/huang2013/catalog/huang2013_cgs_model.txt \
    -o inputs/huang2013/models/huang2013_models.yaml \
    -v
```
