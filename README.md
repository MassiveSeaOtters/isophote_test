# MockGal - Mock Galaxy Image Generator

A Python tool for generating realistic mock galaxy images with Sersic profiles, designed for testing isophote fitting algorithms and galaxy photometry pipelines.

## Features

- Multi-component Sersic profile rendering
- Support for libprofit (fast C++) and astropy (pure Python) backends
- PSF convolution (Gaussian, Moffat, or custom image)
- Realistic sky background (flat or tilted polynomial)
- Noise injection (fixed sigma or SNR-based)
- Batch processing with optional parallelization
- FITS output with complete metadata headers

## Requirements

```bash
pip install numpy scipy astropy pyyaml
```

For optimal performance, build libprofit and ensure `profit-cli` is available (set `LIBPROFIT_PATH` or the legacy `PROFIT_CLI_PATH`):
```bash
export LIBPROFIT_PATH=/path/to/profit-cli
```

## Quick Start

### Single Galaxy Mode

Generate a simple de Vaucouleurs galaxy:

```bash
python mockgal.py --single \
    -z 0.05 \
    --r-eff 5.0 \
    --abs-mag -21.0 \
    --sersic-n 4.0 \
    --psf --psf-fwhm 0.8 \
    --snr 100 \
    -o output/
```

Two-component bulge+disk system:

```bash
python mockgal.py --single \
    --name "bulge_disk" \
    -z 0.03 \
    --r-eff 1.0 3.0 \
    --abs-mag -19.0 -20.5 \
    --sersic-n 4.0 1.0 \
    --ellip 0.2 0.5 \
    --pa 45 45 \
    --psf --psf-fwhm 1.0 \
    --snr 50 \
    -o output/
```

### Direct API Usage

Generate a mock image directly from Python:

```python
from mockgal import SersicComponent, ImageConfig, generate_mock_image

components = [
    SersicComponent(r_eff_kpc=1.0, abs_mag=-20.0, n=4.0, ellipticity=0.2, pa_deg=30.0)
]
config = ImageConfig(size_pixels=51, engine="auto")

image, metadata = generate_mock_image(
    name="api_demo",
    redshift=0.01,
    components=components,
    config=config,
)
```

Generate a mock image from a model file:

```python
from mockgal import generate_mock_image_from_model

image, metadata = generate_mock_image_from_model(
    model_path="examples/example_models.yaml",
    galaxy_name="NGC_1399",
    config={"size_pixels": 51, "engine": "auto"},
)
```

### Batch Mode with Model Files

Process multiple galaxies from a YAML model file:

```bash
python mockgal.py \
    --models examples/huang2013_models.yaml \
    --config examples/example_image_config.yaml \
    -o output/ \
    --workers 1
```

Select specific galaxies:

```bash
python mockgal.py \
    --models examples/huang2013_models.yaml \
    --config examples/example_image_config.yaml \
    --galaxy "NGC 1399" "IC 1459" "NGC 1407" \
    -o output/ \
    --workers 4
```

## Input Files

### Model File (YAML/JSON)

Define galaxies with multiple Sersic components:

```yaml
galaxies:
  - name: NGC_1399
    redshift: 0.01
    components:
      - r_eff_kpc: 0.9
        abs_mag: -20.26
        n: 1.45
        ellipticity: 0.17
        pa_deg: 116.8
      - r_eff_kpc: 10.93
        abs_mag: -22.01
        n: 3.13
        ellipticity: 0.08
        pa_deg: 88.63
```

### Image Config File (YAML/JSON)

Define image generation settings:

```yaml
image_configs:
  - name: "realistic"
    pixel_scale: 0.3          # arcsec/pixel
    zeropoint: 27.0
    engine: libprofit         # or astropy, auto
    psf_enabled: true
    psf_type: moffat
    psf_fwhm: 0.8
    psf_moffat_beta: 4.765
    noise_enabled: true
    noise_snr: 50
    noise_seed: 42
```

## Huang 2013 Catalog

The repository includes the Huang et al. (2013) CGS Survey catalog with 93 nearby elliptical galaxies:

```bash
# Convert ASCII catalog to YAML
python examples/convert_huang2013.py examples/huang2013_cgs_model.txt -o examples/huang2013_models.yaml

# Generate images for selected galaxies
python mockgal.py \
    --models examples/huang2013_models.yaml \
    --galaxy "IC 1459" "NGC 1399" "NGC 1407" \
    --config examples/huang2013_test_config.yaml \
    -o output/huang2013_test/ \
    --workers 1
```

## Command Line Reference

```
usage: mockgal.py [-h] (--models FILE | --single) [--config FILE]
                  [--name NAME] [-z REDSHIFT] [--r-eff KPC [KPC ...]]
                  [--abs-mag MAG [MAG ...]] [--sersic-n N [N ...]]
                  [--ellip ELLIP [ELLIP ...]] [--pa PA [PA ...]]
                  [--pixel-scale PIXEL_SCALE] [--zeropoint ZEROPOINT]
                  [--size-factor SIZE_FACTOR] [--size PIXELS]
                  [--psf] [--psf-fwhm PSF_FWHM] [--psf-type {gaussian,moffat,image}]
                  [--psf-file FILE] [--moffat-beta MOFFAT_BETA]
                  [--sky LEVEL] [--sky-tilted COEFF [COEFF ...]]
                  [--noise-sigma SIGMA] [--snr SNR] [--seed SEED]
                  [--engine {libprofit,astropy,auto}] [--profit-cli PATH]
                  [-o DIR] [--format {fits,npy}]
                  [--galaxy NAME [NAME ...]] [--workers WORKERS] [-v]
```

### Key Options

| Option | Description |
|--------|-------------|
| `--models FILE` | Input model file (YAML/JSON) with galaxy definitions |
| `--single` | Single galaxy mode (parameters via CLI) |
| `--config FILE` | Image config file (PSF, sky, noise settings) |
| `--galaxy NAME` | Select specific galaxies from model file |
| `--workers N` | Number of parallel workers (default: 8, use 1 for sequential) |
| `--engine` | Rendering engine: `libprofit`, `astropy`, or `auto` |
| `--size PIXELS` | Fixed image size (overrides size_factor) |
| `-o DIR` | Output directory |
| `-v` | Verbose output |
| `--sky-sb-value MAG` | Sky surface brightness (mag/arcsec^2) for sky background |
| `--sky-sb-limit MAG` | 5-sigma surface brightness limit (mag/arcsec^2) for Gaussian noise |
| `--gain GAIN` | Detector gain (e-/ADU) for Poisson noise |

## Output

FITS files include comprehensive headers with:
- Galaxy parameters (name, redshift, components)
- Image settings (pixel scale, zeropoint, size)
- PSF configuration (type, FWHM, beta)
- Noise parameters (sigma, SNR, seed)
- Component details (Re, magnitude, Sersic n, ellipticity, PA)

## Notes

### Image Size Limits

To prevent memory issues, image dimensions are automatically capped at 4001 pixels. A warning is displayed when this limit is applied:

```
WARNING - Computed image size 9171x9171 exceeds maximum (4001). Capping to 4001x4001 pixels.
```

### libprofit PSF Convolution

On some systems, `profit-cli` fails to load PSF FITS files (e.g., "less data found than expected"). In this case, PSF convolution is applied in Python after the libprofit render, which keeps results consistent across engines.

### Background-Dominated Noise

Two background-dominated noise modes are supported:

1. **SB limit (Gaussian)**: Provide a 5-sigma surface brightness limit (`sky_sb_limit`, mag/arcsec^2). The per-pixel sigma is derived as:
   `sigma = 10**(-0.4 * (sky_sb_limit - zeropoint)) * pixel_scale**2 / 5`
2. **Sky SB + gain (Poisson)**: Provide `sky_sb_value` (mag/arcsec^2) and `gain` (e-/ADU). A flat sky image is added to the galaxy, then Poisson noise is drawn on the combined image.

If both `sky_sb_value` and `sky_sb_limit` are provided, `sky_sb_value` takes precedence (a warning is logged).

### Memory-Limited Systems

For systems with limited memory:
- Use `--workers 1` to disable parallelization
- Use `--size PIXELS` to set a fixed (smaller) image size
- Process galaxies in smaller batches

### Filename Convention

Output filenames replace spaces with underscores:
- Galaxy "IC 1459" with config "clean" → `IC_1459_clean.fits`

## References

- Huang S., Ho L.C., Peng C.Y., Li Z.-Y., Barth A.J. (2013), ApJ, 766, 47
- libprofit: https://github.com/ICRAR/libprofit
