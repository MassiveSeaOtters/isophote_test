# MockGal

MockGal generates mock galaxy images for isophote-fitting tests and related photometry workflows. It renders multi-component galaxies (Sersic + Ferrer bar + PSF-convolved nucleus) via a polymorphic `Component` ABC, with three rendering backends:

- `libprofit` (preferred): native rendering of all profile types via `profit-cli`.
- `astropy` (fallback): pure-Python rendering for Sersic and PointSource; Ferrer raises `NotImplementedError`.
- GALFIT binary via `mockgal_galfit.py`: pixel-perfect reference renders, used for cross-validation.

## Repository Layout

```text
mockgal.py
mockgal_galfit.py
inputs/
tests/
benchmarks/
docs/
output/
```

- `mockgal.py`: CLI entry point and core library
- `mockgal_galfit.py`: GALFIT-backed reference renderer
- `inputs/`: canonical Huang2013 assets, examples, and demos
- `tests/`: pytest suite (228 passing)
- `benchmarks/`: performance benchmarks and reports
- `docs/`: lessons, build notes, quick references, and planning docs
- `output/`: generated artifacts, not tracked in git

## Setup

Install dependencies with `uv`. This repo does not yet ship a lockfile or `pyproject.toml`, so use `uv pip` in your existing environment:

```bash
uv pip install numpy scipy astropy pyyaml pytest
```

### libprofit (optional but preferred)

MockGal prefers the `libprofit` backend via `profit-cli`. To enable it, point MockGal
at your local `profit-cli` binary with **one** of the following:

```bash
# Preferred: directory that contains profit-cli, or the binary itself
export LIBPROFIT_PATH=/absolute/path/to/libprofit/build

# Alternative env var (same semantics)
export PROFIT_CLI_PATH=/absolute/path/to/libprofit/build/profit-cli

# Or pass it on the CLI per invocation
python mockgal.py --profit-cli-path /absolute/path/to/libprofit/build/profit-cli ...
```

> **IMPORTANT: do not hardcode your local `LIBPROFIT_PATH` inside `mockgal.py`.**
> Earlier versions of this repo accidentally overwrote `os.environ["LIBPROFIT_PATH"]`
> at module import time with a specific developer's path. That silently disabled the
> documented `export LIBPROFIT_PATH=...` workflow for everyone else, forced the
> fallback to the `astropy` engine on machines where the hardcoded path did not exist,
> and mutated the process-wide `PATH` as a side effect. Keep user-specific paths in
> your shell environment (or pass `--profit-cli-path`), never in tracked source.

If `profit-cli` cannot be located (or fails its dynamic-library health check),
MockGal falls back to the pure-Python `astropy` engine in `auto` mode and logs a
warning.

## Quick Start

Single galaxy from the CLI:

```bash
python mockgal.py --single \
    --name de_vaucouleurs \
    -z 0.05 \
    --r-eff 5.0 \
    --abs-mag -21.0 \
    --sersic-n 4.0 \
    --psf --psf-fwhm 0.8 \
    --sky-sb-limit 27.0 \
    -o output/
```

Batch run with canonical Huang2013 assets:

```bash
python mockgal.py \
    --models inputs/huang2013/models/huang2013_models.yaml \
    --config inputs/huang2013/configs/huang2013_test_config.yaml \
    --galaxy "IC 1459" "NGC 1399" "NGC 1407" \
    --workers 1 \
    -o output/huang2013_test
```

Systematic Huang2013 mocks:

```bash
python inputs/huang2013/scripts/generate_huang2013_mocks.py \
    --run-manifest inputs/huang2013/runs/huang2013_production_baseline.yaml \
    --output output/huang2013_production/huang2013_production_baseline \
    --test
```

## Direct API Usage

```python
from mockgal import ImageConfig, SersicComponent, generate_mock_image

components = [
    SersicComponent(r_eff_kpc=1.0, abs_mag=-20.0, n=4.0, ellipticity=0.2, pa_deg=30.0)
]

image, metadata = generate_mock_image(
    name="api_demo",
    redshift=0.01,
    components=components,
    config=ImageConfig(size_pixels=51, engine="auto"),
    return_metadata=True,
)

# Rectangular image: size_pixels can be (ny, nx) tuple
config_rect = ImageConfig(size_pixels=(100, 150), engine="auto")
image_rect, metadata_rect = generate_mock_image(
    name="api_demo_rect",
    redshift=0.01,
    components=components,
    config=config_rect,
)
```

### Multi-component galaxies (bulge + bar + nucleus)

```python
from mockgal import (
    ImageConfig,
    SersicComponent,
    FerrerComponent,
    PointSourceComponent,
    generate_mock_image,
)

components = [
    SersicComponent(r_eff_kpc=0.8, abs_mag=-19.0, n=4.0,
                    ellipticity=0.1, pa_deg=0.0),     # bulge
    SersicComponent(r_eff_kpc=4.0, abs_mag=-20.0, n=1.0,
                    ellipticity=0.4, pa_deg=30.0),    # disk
    FerrerComponent(r_out_kpc=2.5, abs_mag=-18.5,
                    alpha=2.0, beta=0.0,
                    ellipticity=0.6, pa_deg=15.0),    # bar (libprofit only)
    PointSourceComponent(abs_mag=-16.0),               # nucleus (requires PSF)
]

image, metadata = generate_mock_image(
    name="multi_component_demo",
    redshift=0.01,
    components=components,
    config=ImageConfig(size_pixels=201, engine="libprofit",
                       psf_enabled=True, psf_type="gaussian", psf_fwhm=1.0),
)
```

`PointSourceComponent` requires `psf_enabled=True` (rendering a delta without a PSF would produce a single bright pixel). `FerrerComponent` is libprofit-only — the astropy backend raises `NotImplementedError`.

### GALFIT-backed reference render

For pixel-perfect parity with the original GALFIT integrator (e.g. when validating a parsed Salo+2015 P4 model), use `mockgal_galfit.py`:

```python
from mockgal import MockGalaxy, SersicComponent, ImageConfig
from mockgal_galfit import render_with_galfit

galaxy = MockGalaxy(name="ref", redshift=0.01,
                    components=[SersicComponent(r_eff_kpc=2.0, abs_mag=-19.5, n=4.0)])
config = ImageConfig(size_pixels=101, pixel_scale=0.168, zeropoint=27.0,
                     psf_enabled=True, psf_type="gaussian", psf_fwhm=1.0)
image, meta = render_with_galfit(galaxy, config)
```

By default the GALFIT binary is `/Users/shuang/code/galfit/galfit`; override via the `GALFIT_BIN` env var or the `galfit_bin=` kwarg.

## Huang2013 Notes

- The canonical Huang2013 inputs live under `inputs/huang2013/`.
- `inputs/huang2013/models/huang2013_models.yaml` contains the 93-galaxy model set.
- `inputs/huang2013/scripts/generate_huang2013_mocks.py` now runs manifest-defined Huang2013 production batches from `inputs/huang2013/runs/`.
- The baseline production manifest keeps HSC-like settings with `pixel_scale = 0.168`, `psf_fwhm = 0.7`, `size_factor = 6` anchored on Huang2013 `re_overall`, an explicit `max_image_size = 4001`, and baseline noisy rows at `sky_sb_limit = 24.5` plus a low-noise reference row at `29.0`.
- `inputs/huang2013/configs/huang2013_hsc_i_calibration.yaml` stores optional Huang2013-only HSC `i`-band `wide` and `dud` depth references; those values are not global defaults.
- Use small validation runs before attempting a broad batch job.

## Development

Run tests:

```bash
pytest tests/test_mockgal.py -v
```

Run benchmarks:

```bash
python benchmarks/bench_engines.py
```

## Documentation

- `AGENTS.md`: repo workflow rules
- `CLAUDE.md`: concise project map for Claude-compatible tools
- `docs/LESSON.md`: durable lessons and pitfalls
- `inputs/README.md`: canonical input inventory and Huang2013 table of contents
- `docs/LIBPROFIT_COMPILE.md`: local `libprofit` build notes
- `docs/PROFIT_CLI_USAGE.md`: raw `profit-cli` help text
