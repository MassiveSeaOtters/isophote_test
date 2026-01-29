# Changes Summary

## Directory Reorganization

All input files have been moved to the `examples/` directory for better organization:

### Files Moved to `examples/`
- `convert_huang2013.py` → `examples/convert_huang2013.py`
- `huang2013_models.yaml` → `examples/huang2013_models.yaml`
- `huang2013_test_config.yaml` → `examples/huang2013_test_config.yaml`
- `example_image_config.yaml` → `examples/example_image_config.yaml`
- `example_models.yaml` → `examples/example_models.yaml`
- `huang2013_cgs_model.txt` → `examples/huang2013_cgs_model.txt`

### Updated Files
- `tests/test_mockgal.py` - Updated path to `examples/huang2013_cgs_model.txt`
- `README.md` - Updated all example commands to reference `examples/` directory
- `CLAUDE.md` - Updated file paths and added visualization documentation

## New Visualization Function

Added `visualize_galaxy()` function to `mockgal.py` for creating publication-quality visualizations:

### Features
- **Arcsinh scaling**: Handles wide dynamic range of galaxy images
- **Perceptually uniform colormaps**: Supports viridis, magma, inferno, cividis, plasma
- **Smoothed contours**: Overlays contours to highlight galaxy morphology
- **Automatic output**: PNG file generated with same prefix as input FITS
- **Flexible input**: Accepts FITS file path or numpy array + metadata
- **Customizable**: Control colormap, smoothing, contour levels, DPI, figure size

### Usage Examples

```python
from mockgal import visualize_galaxy

# Basic usage (auto-generates PNG with same prefix)
visualize_galaxy('output/NGC_1399_clean.fits')

# Custom colormap
visualize_galaxy('output/IC_1459_clean.fits', cmap='magma')

# Full customization
visualize_galaxy('output/galaxy.fits',
                output_path='custom_output.png',
                cmap='inferno',
                sigma_smooth=3.0,
                n_contours=10,
                figsize=(12, 12),
                dpi=300)

# From array
visualize_galaxy(image, metadata=meta, output_path='galaxy.png')
```

### Dependencies
Requires matplotlib (optional):
```bash
pip install matplotlib
```

## New Utility Function

Added `parse_huang2013()` function to `mockgal.py`:
- Parses Huang 2013 ASCII catalog directly
- Returns dictionary mapping galaxy names to lists of SersicComponent objects
- Primarily for testing purposes
- Production use should convert to YAML first using `examples/convert_huang2013.py`

## Demo Script

Created `demo_visualization.py` to demonstrate the complete workflow:
- Generates a two-component mock galaxy (bulge + disk)
- Saves FITS file
- Creates visualizations with multiple colormaps

Run with:
```bash
python demo_visualization.py
```

## Testing

All tests pass except for 2 pre-existing failures unrelated to these changes:
- `test_engine_selection_auto` - naming inconsistency (libprofit vs pyprofit)
- `test_psf_smoothing_effect` - PSF behavior test

Run tests with:
```bash
pytest tests/test_mockgal.py -v
```

## Updated Documentation

### CLAUDE.md
- Updated all file paths to reference `examples/` directory
- Added visualization function documentation
- Added new functions to "Key Functions" section
- Updated example commands

### README.md
- Updated example commands to use `examples/` directory paths

## Migration Guide

If you have existing scripts that reference the old file locations, update them:

```bash
# Old
python mockgal.py --models huang2013_models.yaml --config example_image_config.yaml

# New
python mockgal.py --models examples/huang2013_models.yaml --config examples/example_image_config.yaml
```

Similarly for `convert_huang2013.py`:

```bash
# Old
python convert_huang2013.py huang2013_cgs_model.txt -o output.yaml

# New
python examples/convert_huang2013.py examples/huang2013_cgs_model.txt -o examples/output.yaml
```
