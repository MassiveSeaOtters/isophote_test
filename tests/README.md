# Tests Directory

This directory contains the test suite for MockGal using pytest.

## Test Coverage

The test suite includes **11 test classes** with **57+ test functions** covering:

### Test Classes

1. **TestCosmology** - Cosmological distance and magnitude calculations
   - Luminosity distance at various redshifts
   - Absolute to apparent magnitude conversion
   - Distance modulus calculations

2. **TestSurfaceBrightnessConversions** - Surface brightness transformations
   - Sersic profile surface brightness calculations
   - Central surface brightness from total magnitude
   - Flux to magnitude conversions

3. **TestSersicComponent** - Sersic profile component validation
   - Component initialization and validation
   - Parameter bounds checking (Sersic index, ellipticity)
   - Error handling for invalid parameters

4. **TestMockGalaxy** - Multi-component galaxy models
   - Single and multi-component galaxies
   - Effective radius calculations
   - YAML/JSON serialization

5. **TestImageConfig** - Image generation configuration
   - PSF configuration (Gaussian, Moffat)
   - Sky background and noise settings
   - Configuration validation and defaults

6. **TestSersicEngine** - Rendering backend tests
   - Astropy backend rendering
   - LibProfit backend rendering (if available)
   - PSF convolution
   - Elliptical profile generation

7. **TestMockImageGenerator** - End-to-end image generation pipeline
   - Single component rendering
   - Multi-component rendering
   - Coordinate transformations
   - Metadata generation
   - Error handling

8. **TestCLI** - Command-line interface
   - Single galaxy mode
   - Batch processing mode
   - Argument parsing and validation

9. **TestHuangParser** - Huang et al. (2013) catalog parsing
   - ASCII catalog parsing
   - YAML conversion
   - Data validation

10. **TestOutput** - File output operations
    - FITS file generation and metadata
    - NPY file generation with JSON sidecar
    - Output directory creation

11. **TestConventions** - Naming and convention compliance
    - Filename sanitization
    - Output path generation

## Running Tests

Run the full test suite:
```bash
pytest tests/test_mockgal.py -v
```

Run specific test class:
```bash
pytest tests/test_mockgal.py::TestSersicEngine -v
```

Run specific test function:
```bash
pytest tests/test_mockgal.py::TestOutput::test_save_fits -v
```

Run with coverage report:
```bash
pytest tests/test_mockgal.py --cov=mockgal --cov-report=html
```

## Test Output Organization

Tests that generate images save outputs to organized subdirectories:

```
output/
├── test_cli/              # CLI test outputs
├── test_generator/        # Generator test outputs
├── test_output/           # Output format test outputs
└── test_visualization/    # Visualization test outputs
```

These directories are automatically created during test runs and are not tracked by git.

## Test Fixtures

Common fixtures defined in `test_mockgal.py`:

- `simple_galaxy` - Single-component Sersic galaxy for basic tests
- `default_config` - Standard image configuration
- `huang_catalog_path` - Path to Huang et al. (2013) catalog file
- `tmp_path` - Pytest built-in temporary directory

## Dependencies

Tests require:
- `pytest` - Test framework
- `astropy` - Astronomy library (core rendering engine)
- `numpy` - Numerical operations
- `pyyaml` - YAML parsing

Optional for enhanced testing:
- `libprofit` - Fast C++ rendering backend
- `pytest-cov` - Coverage reporting
- `scipy` - PSF convolution tests
