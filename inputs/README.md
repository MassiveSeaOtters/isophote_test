# Examples Directory

This directory contains input data files (galaxy models and image configurations) and utility scripts for MockGal.

> **Note:** This directory will be renamed to `inputs/` to better reflect its purpose as input data rather than example code. See `MIGRATION.md` for details.

## Directory Contents

### Model Files

Galaxy model definitions in YAML format:

| File | Description | Count |
|------|-------------|-------|
| `example_models.yaml` | Simple example galaxies for testing | 4 galaxies |
| `huang2013_models.yaml` | Huang et al. (2013) catalog galaxies | 93 galaxies |

**Model file format:**
```yaml
- name: "NGC 3923"
  redshift: 0.0059
  components:
    - r_eff: 5.0
      abs_mag: -21.5
      n: 4.0
      ellipticity: 0.2
      position_angle: 45.0
```

### Configuration Files

Image generation settings in YAML format:

| File | Description | Configs |
|------|-------------|---------|
| `example_image_config.yaml` | Various PSF, noise, sky settings | 6 configs |
| `huang2013_test_config.yaml` | Test config for Huang2013 galaxies | 1 config |

**Config file format:**
```yaml
- name: "clean"
  pixel_scale: 0.3
  zeropoint: 27.0
  engine: libprofit
  psf_enabled: false
  sky_enabled: false
  noise_enabled: false
```

### Catalog Data

| File | Description | Source |
|------|-------------|--------|
| `huang2013_cgs_model.txt` | ASCII catalog from Huang et al. (2013) | [ADS Link](https://ui.adsabs.harvard.edu/abs/2013ApJ...766...47H) |

Original data format with columns: Name, Distance, VMag, Re_V, etc.

### Utility Scripts

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `convert_huang2013.py` | Convert ASCII catalog to YAML | `huang2013_cgs_model.txt` | `huang2013_models.yaml` |
| `api_call_demo.py` | Demonstrate Python API usage | - | Demo images |
| `demo_visualization.py` | Demonstrate visualization features | Model + config files | PNG visualizations |
| `huang2013_noise_sblimit_demo.py` | Demo sky background and noise simulation | Huang2013 models | Noisy galaxy images |
| `huang2013_noise_sbvalue_demo.py` | Demo noise at specific surface brightness | Huang2013 models | Noisy galaxy images |

## Usage Examples

### Using Model Files

**Single galaxy from catalog:**
```bash
python mockgal.py \
    --models examples/huang2013_models.yaml \
    --config examples/example_image_config.yaml \
    --galaxy "NGC 3923" \
    -o output/
```

**Batch processing multiple galaxies:**
```bash
python mockgal.py \
    --models examples/huang2013_models.yaml \
    --config examples/huang2013_test_config.yaml \
    --galaxy "NGC 1399" "NGC 1407" "IC 1459" \
    --workers 1 \
    -o output/batch/
```

### Converting Catalog Data

Regenerate YAML from ASCII source:
```bash
python examples/convert_huang2013.py \
    examples/huang2013_cgs_model.txt \
    -o examples/huang2013_models_new.yaml \
    -v
```

### Running Demo Scripts

**API usage demo:**
```bash
python examples/api_call_demo.py
```

**Visualization demo:**
```bash
python examples/demo_visualization.py
```

**Noise simulation demos:**
```bash
python examples/huang2013_noise_sblimit_demo.py
python examples/huang2013_noise_sbvalue_demo.py
```

## File Relationships

```
huang2013_cgs_model.txt  (source ASCII catalog)
         |
         v
convert_huang2013.py  (conversion script)
         |
         v
huang2013_models.yaml  (YAML galaxy models)
         |
         v
mockgal.py --models ... --config ...  (image generation)
         |
         v
output/*.fits  (mock galaxy images)
```

## Data Sources and Citations

### Huang et al. (2013) Catalog

**Citation:**
> Huang, Z., Radburn-Smith, D. J., De Jong, R. S., et al. 2013, ApJ, 766, 47
> "The Opacity of Spiral Galaxy Disks. VII. The Accuracy of Galaxy Counts as an Extinction Probe"

**Data characteristics:**
- 93 nearby galaxies from Carnegie-Irvine Galaxy Survey (CGS)
- V-band photometry with Sersic fits
- Absolute magnitudes (distance modulus already applied)
- Typical redshift: z ≈ 0.01
- Magnitude range: -17 to -23 (absolute V-band)

## Adding New Data

### Creating a New Model File

1. Create YAML file with galaxy definitions
2. Include required fields: name, redshift, components
3. Each component needs: r_eff, abs_mag, n
4. Optional fields: ellipticity, position_angle

Example:
```yaml
- name: "My Galaxy"
  redshift: 0.01
  components:
    - r_eff: 10.0        # arcsec
      abs_mag: -21.0     # absolute magnitude
      n: 2.5             # Sersic index
      ellipticity: 0.3   # 1 - b/a
      position_angle: 30 # degrees from +Y axis
```

### Creating a New Config File

Define image generation settings:
```yaml
- name: "my_config"
  pixel_scale: 0.3      # arcsec/pixel
  zeropoint: 27.0       # magnitude zeropoint
  engine: libprofit     # or "astropy"
  psf_enabled: true
  psf_type: gaussian    # or "moffat"
  psf_fwhm: 1.0        # arcsec
  sky_enabled: true
  sky_background: 1000  # counts
  noise_enabled: true
  noise_type: gaussian
  noise_stddev: 10      # counts
```

## Notes

- **Huang2013 magnitudes are already absolute** - Do not apply distance modulus
- **Memory warning**: Processing all 93 Huang2013 galaxies requires significant RAM
  - Use `--galaxy` flag to process subset
  - Use `--workers 1` to process sequentially
- **Output filenames**: Spaces in galaxy names are replaced with underscores (NGC 3923 → NGC_3923.fits)
