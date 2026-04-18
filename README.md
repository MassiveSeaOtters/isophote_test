# MockGal

MockGal generates mock galaxy images with Sersic profiles for isophote-fitting tests and related photometry workflows. It supports both the `libprofit` backend and a pure-Python `astropy` fallback.

## Repository Layout

```text
mockgal.py
inputs/
tests/
benchmarks/
docs/
output/
```

- `mockgal.py`: CLI entry point and core library
- `inputs/`: canonical Huang2013 assets, examples, and demos
- `tests/`: pytest suite
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
