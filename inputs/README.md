# Inputs Directory

This directory contains input data files (galaxy models and image configurations) and utility scripts for MockGal.

> **Note:** This directory was renamed from `examples/` to `inputs/` to better reflect its purpose as input data rather than example code. See `../MIGRATION.md` for migration details.

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
| `generate_huang2013_mocks.py` | **Generate systematic mock images for all 93 Huang2013 galaxies** | `huang2013_models.yaml` | 372 FITS files (4 mocks × 93 galaxies) |
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

### Systematic Mock Generation for Huang2013 Sample

The `generate_huang2013_mocks.py` script creates a complete set of mock images for all 93 Huang2013 galaxies with standardized configurations. This is useful for systematic testing of isophote fitting algorithms at different redshifts and noise levels.

#### Mock Configurations

The script generates **4 mock versions** of each galaxy:

| Mock | Redshift | Pixel Scale | PSF | Noise | Purpose |
|------|----------|-------------|-----|-------|---------|
| **mock1** | z=0.05 | HSC (0.168"/pix) | FWHM=0.7" | None | Clean, nearby galaxy baseline |
| **mock2** | z=0.05 | HSC (0.168"/pix) | FWHM=0.7" | sky_sb_limit=24.0 | Nearby with realistic noise |
| **mock3** | z=0.20 | HSC (0.168"/pix) | FWHM=0.7" | sky_sb_limit=24.0 | Intermediate redshift |
| **mock4** | z=0.50 | HSC (0.168"/pix) | FWHM=0.7" | sky_sb_limit=24.0 | High redshift, fainter |

**What each mock means:**

- **Mock 1** (z=0.05, no noise): Best-case scenario for nearby galaxies. Use this to establish ground truth for isophote fitting algorithms without noise complications.

- **Mock 2** (z=0.05, noisy): Realistic nearby galaxy observation with HSC-like depth (24 mag/arcsec² at 5σ). Tests algorithm robustness to noise at high S/N.

- **Mock 3** (z=0.20, noisy): Galaxies appear smaller and fainter (~4× lower surface brightness than mock2). Tests performance on intermediate-redshift galaxies where surface brightness dimming becomes significant.

- **Mock 4** (z=0.50, noisy): Challenging regime where galaxies are significantly smaller and fainter (~20× lower surface brightness than mock2). Tests algorithm limits and determines the effective redshift range for reliable fitting.

**Key Parameters:**
- **HSC pixel scale**: 0.168 arcsec/pixel (Hyper Suprime-Cam on Subaru Telescope)
- **PSF**: Gaussian with FWHM=0.7 arcsec (typical good seeing)
- **Image size**: Maximum 4000×4000 pixels (enforced to prevent memory issues)
- **Noise model**: Sky-background-dominated Gaussian noise (sky_sb_limit = 5σ depth)
- **Zeropoint**: 27.0 mag (typical for modern surveys)

#### Usage

**Generate all 93 galaxies (recommended for full datasets):**
```bash
python inputs/generate_huang2013_mocks.py --output /path/to/output
```

**Generate specific galaxies for testing:**
```bash
python inputs/generate_huang2013_mocks.py --output /path/to/output \
    --galaxies "NGC 3923" "IC 1459" "NGC 1399"
```

**Test with first 2 galaxies:**
```bash
python inputs/generate_huang2013_mocks.py --output /path/to/output --test
```

**Get help:**
```bash
python inputs/generate_huang2013_mocks.py --help
```

#### Output Structure

```
output_dir/
└── huang2013/
    ├── ESO185-G054/
    │   ├── ESO185-G054_mock1.fits  # z=0.05, no noise
    │   ├── ESO185-G054_mock2.fits  # z=0.05, with noise
    │   ├── ESO185-G054_mock3.fits  # z=0.20, with noise
    │   └── ESO185-G054_mock4.fits  # z=0.50, with noise
    ├── NGC3923/
    │   ├── NGC3923_mock1.fits
    │   ├── NGC3923_mock2.fits
    │   ├── NGC3923_mock3.fits
    │   └── NGC3923_mock4.fits
    └── ... (91 more galaxies)
```

Each galaxy gets its own subdirectory with 4 FITS files (one per mock configuration).

#### Performance and Output Size

- **Processing time**: ~23 seconds per galaxy × 4 mocks = ~92 sec/galaxy
- **Full dataset**: 93 galaxies × ~92 sec ≈ **35 minutes**
- **File size**: ~61 MB per FITS file (4000×4000 pixels)
- **Per-galaxy output**: 4 files × 61 MB ≈ **244 MB**
- **Total output**: 372 files ≈ **23 GB**

#### Use Cases

1. **Algorithm benchmarking**: Compare isophote fitting performance across the 4 redshifts
2. **Noise robustness testing**: Mock1 vs Mock2 shows impact of noise at fixed redshift
3. **Redshift scaling**: Mock2/3/4 show how galaxy size and S/N change with redshift
4. **Training data**: Use for machine learning algorithms that need diverse galaxy images
5. **Systematic validation**: Ensure isophote fitting works across the full Huang2013 sample

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
