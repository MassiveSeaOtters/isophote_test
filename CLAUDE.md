# CLAUDE.md - Development Guidelines for MockGal

This file contains guidelines and context for AI assistants (Claude) working on this codebase.

## Project Overview

MockGal is a mock galaxy image generator for testing isophote fitting algorithms. It uses Sersic profiles rendered via libprofit (C++) or astropy (Python) backends.

### Key Files

| File | Purpose |
|------|---------|
| `mockgal.py` | Main generator script (CLI + library) |
| `examples/convert_huang2013.py` | Converts Huang 2013 ASCII catalog to YAML |
| `examples/huang2013_models.yaml` | 93 galaxies from Huang et al. (2013) |
| `examples/example_image_config.yaml` | Sample image configurations |
| `examples/huang2013_test_config.yaml` | Test config for Huang2013 galaxies |
| `examples/huang2013_cgs_model.txt` | Source ASCII catalog from Huang et al. (2013) |
| `examples/example_models.yaml` | Example galaxy models for testing |
| `tests/test_mockgal.py` | Test suite |
| `benchmarks/bench_engines.py` | Performance benchmarks |

## Ground Rules

### Memory Management

1. **Never run the full Huang2013 catalog (93 galaxies) in batch mode on memory-limited machines**
   - Test with only 4-5 galaxies: `--galaxy "IC 1459" "NGC 1399" "NGC 1407" "NGC 1404"`

2. **Disable parallelization on memory-limited systems**
   - Always use `--workers 1` for sequential processing
   - Default workers=8 can cause memory issues with large images

3. **Image size cap at 4001 pixels**
   - `MAX_IMAGE_SIZE = 4001` in mockgal.py
   - Automatic warning and fallback when exceeded
   - Do not increase this limit without explicit user request

### Data Conventions

4. **Huang 2013 magnitudes are already absolute**
   - The VMag column in `huang2013_cgs_model.txt` is absolute V-band magnitude
   - Do NOT apply distance modulus conversion
   - Default redshift for Huang2013 galaxies: z=0.01

5. **Output filenames must not contain spaces**
   - Replace spaces with underscores: "IC 1459" → "IC_1459"
   - Applied in `process_single_job()` and single-galaxy mode

### Testing

6. **Standard test command for Huang2013 batch mode:**
   ```bash
   python mockgal.py \
       --models examples/huang2013_models.yaml \
       --config examples/huang2013_test_config.yaml \
       --galaxy "ESO 185-G054" "ESO 221-G026" "IC 1459" "IC 1633" "IC 2006" \
       --workers 1 \
       -o output/huang2013_test \
       -v
   ```

7. **Verify output with:**
   ```bash
   python -c "
   from astropy.io import fits
   f = fits.open('output/huang2013_test/IC_1459_clean.fits')
   print(f'Shape: {f[0].data.shape}')
   print(f'Max flux: {f[0].data.max():.4e}')
   "
   ```

8. All examples or demos should go into the "examples" folder.

## Code Structure

### Main Classes

- `SersicComponent`: Single Sersic profile parameters
- `MockGalaxy`: Galaxy with multiple components
- `ImageConfig`: Image generation settings (PSF, sky, noise)
- `SersicEngine`: Abstraction for libprofit/astropy backends
- `MockImageGenerator`: Main image generation pipeline

### Key Functions

- `load_model_file()`: Parse galaxy definitions from YAML/JSON
- `load_image_configs()`: Parse image settings from YAML/JSON
- `run_batch()`: Batch processing with optional parallelization
- `process_single_job()`: Process one galaxy+config combination
- `save_fits()`: Save image with metadata headers
- `save_npy()`: Save image as numpy array with JSON metadata
- `generate_mock_image()`: Convenience API for direct image generation
- `visualize_galaxy()`: Create PNG visualization with arcsinh scaling and contours
- `parse_huang2013()`: Parse Huang 2013 ASCII catalog (for testing)

### Constants

```python
DEFAULT_PIXEL_SCALE = 0.3      # arcsec/pixel
DEFAULT_ZEROPOINT = 27.0       # mag
DEFAULT_SIZE_FACTOR = 15.0     # image half-size = factor * max(Re)
DEFAULT_REDSHIFT = 0.01
MAX_SERSIC_INDEX = 8.0
MAX_IMAGE_SIZE = 4001          # maximum image dimension
```

## Common Tasks

### Adding a New Image Config

Add to `examples/example_image_config.yaml`:
```yaml
- name: "new_config"
  pixel_scale: 0.3
  zeropoint: 27.0
  engine: libprofit
  psf_enabled: true
  psf_type: gaussian
  psf_fwhm: 1.0
```

### Regenerating Huang2013 Models

If the source catalog or conversion logic changes:
```bash
python examples/convert_huang2013.py examples/huang2013_cgs_model.txt -o examples/huang2013_models.yaml -v
```

### Visualizing Mock Galaxy Images

Generate publication-quality visualizations with arcsinh scaling:
```python
from mockgal import visualize_galaxy

# Visualize a FITS file (auto-generates PNG with same prefix)
visualize_galaxy('output/IC_1459_clean.fits')

# Customize visualization
visualize_galaxy('output/NGC_1399_clean.fits',
                cmap='magma',           # perceptually uniform colormap
                sigma_smooth=3.0,       # smoothing for contours
                n_contours=10,          # number of contour levels
                dpi=300)                # high-res output

# Visualize from array
visualize_galaxy(image, metadata=meta, output_path='galaxy.png')
```

The visualization function:
- Uses `np.arcsinh` scaling to handle wide dynamic range
- Supports perceptually uniform colormaps (viridis, magma, inferno, cividis, plasma)
- Overlays smoothed contours to highlight galaxy morphology
- Automatically generates PNG with same filename as input FITS
- Shows galaxy name and configuration in the title

### Running Tests

```bash
pytest tests/test_mockgal.py -v
```

## Troubleshooting

### "profit-cli died with SIGSEGV"
- Image size too large - check if size_factor is creating huge images
- Solution: Use `--size 501` or rely on MAX_IMAGE_SIZE cap

### Extremely high flux values (>1e10)
- Check if magnitudes were incorrectly converted
- Huang2013 VMag is already absolute magnitude (-17 to -23 range)

### Memory errors in batch mode
- Reduce workers: `--workers 1`
- Process fewer galaxies per batch
- Use fixed smaller image size: `--size 1001`

## Coordinate Conventions

- **Position Angle (PA)**: Measured from +Y axis, counter-clockwise (degrees)
- **Ellipticity**: e = 1 - b/a (axis ratio = 1 - ellipticity)
- **Magnitude**: Standard astronomy convention, flux = 10^(-0.4 * (mag - zeropoint))
