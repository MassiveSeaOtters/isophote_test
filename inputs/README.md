# Inputs Directory

This directory contains the canonical input assets for MockGal.

## Structure

```text
inputs/
├── huang2013/
│   ├── catalog/
│   │   └── huang2013_cgs_model.txt
│   ├── configs/
│   │   └── huang2013_test_config.yaml
│   ├── models/
│   │   └── huang2013_models.yaml
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

- `inputs/huang2013/models/huang2013_models.yaml`
- `inputs/huang2013/configs/huang2013_test_config.yaml`
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
    --output output/huang2013_systematic \
    --test
```

Current generator settings:

- HSC-like pixel scale: `0.168` arcsec/pixel
- Gaussian PSF FWHM: `0.7` arcsec
- Dynamic sizing: `size_factor = 16`
- Noisy mocks use `sky_sb_limit = 24.5`
- Maximum image size: `4001`
- One QA mosaic PNG per galaxy

## Examples And Demos

- `inputs/examples/` contains small reusable YAML files for CLI and API examples.
- `inputs/demos/` contains one-off scripts for API usage, visualization, and targeted Huang2013 noise experiments.
- Demo scripts are not part of the canonical Huang2013 workflow.

## File Inventory

| Path | Type | Role | Required For Huang2013 Test Suite |
|---|---|---|---|
| `inputs/huang2013/catalog/huang2013_cgs_model.txt` | ASCII catalog | Source Huang et al. (2013) catalog | No |
| `inputs/huang2013/models/huang2013_models.yaml` | YAML model file | Canonical 93-galaxy Huang2013 model set | Yes |
| `inputs/huang2013/configs/huang2013_test_config.yaml` | YAML config file | Canonical Huang2013 validation config | Yes |
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
