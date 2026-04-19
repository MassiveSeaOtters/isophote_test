# Lessons Learned

## Execution Safety

- Validate Huang2013 workflows on a small subset before attempting a broader batch run.
- On memory-limited machines, run Huang2013 jobs with `--workers 1`; parallel workers can push large images over memory limits.
- Keep the 4001-pixel image-size cap in place unless the user explicitly asks to revisit it.
- In this environment, manually export `LIBPROFIT_PATH=/Users/shuang/Dropbox/work/project/otters/isophote_test/libprofit/mbp` before libprofit-backed Huang2013 validation commands.
- Treat `profit-cli` as available only if it can actually start. A binary that exists on disk but fails its dynamic-library health check should fall back to the astropy engine in `auto` mode.

## Huang2013 Data Conventions

- The Huang2013 `VMag` column is already an absolute V-band magnitude. Do not apply a distance modulus conversion during catalog conversion or mock generation.
- The default Huang2013 redshift assumption is `z = 0.01` for the source models unless a workflow intentionally overrides it for simulated observations.
- One Huang2013 galaxy has `n > 8`; downstream logic clamps extreme Sersic indices, so changes to parser or validation code should preserve that safety behavior.

## Noise And Sky Handling

- Sky should be added internally for Poisson statistics and subtracted before writing the final mock image. Leaving sky in the output breaks the intended sky-free mock convention.
- `sky_sb_value` and `sky_enabled` should remain decoupled. The sky value can be needed for noise modeling even when no sky background should survive in the saved image.
- Randomized noise seeds should be opt-in so science mocks can vary while regression tests remain reproducible.
- For HSC sky-region calibration, request only the image and variance planes. The mask and PSF products are not needed for the blank-sky workflow.
- For HSC `coadd/bg` calibration, center each cutout on its sigma-clipped local median before pooling pixels. Use the pooled centered RMS as the primary mapping into `sky_sb_limit`.
- Use original per-cutout medians only as a secondary summary for exploratory `sky_sb_value` reporting. Coadd-derived `gain` estimates remain too unstable to treat as a detector parameter.
- Keep HSC `i`-band calibration values in Huang2013-specific reference files and require explicit workflow opt-in; do not promote them to repo-wide defaults.
- For Huang2013 production runs, copy calibrated numeric values into run manifests directly rather than referencing calibration profile names at runtime.

## Image Size And Survey Mocking

- Huang2013 image sizing should anchor on galaxy-level `re_overall` rather than the single largest component radius; using the largest component makes the dynamic sizes excessively large for this sample.
- Treat the image-size cap as configuration when reproducibility matters. Recording `max_image_size` in the run manifest is clearer than relying on an implicit global constant.
- The baseline Huang2013 systematic mock workflow uses HSC-like settings: `pixel_scale = 0.168`, `psf_fwhm = 0.7`, and `sky_sb_limit = 24.5`.
- For near-ideal Huang2013 reference images, prefer very low noise such as `sky_sb_limit = 29.0` over perfectly noise-free outputs.
- The canonical Huang2013 production path should be manifest-driven: one YAML file per run family, one row per mock configuration, with explicit numeric values.
- Per-galaxy QA mosaics are useful enough to keep in the canonical Huang2013 generator because they provide quick visual validation without opening four FITS files individually.

## Filename And Output Conventions

- Output filenames must use `sanitize_filename()` so galaxy-name spaces are removed instead of converted to underscores.
- Use a single underscore only between the sanitized galaxy name and the config name, for example `NGC3923_clean.fits`.
- Tests and benchmarks should write into organized `output/` subdirectories rather than cluttering the repo root.

## Documentation Maintenance

- Agent-facing files should reference lessons and workflows rather than duplicating long command catalogs or session history.
- Session-specific status, commit hashes, and handoff notes belong in `docs/journal/` or dedicated session logs, not in `AGENTS.md` or `CLAUDE.md`.
- When the canonical Huang2013 workflow changes, update `inputs/README.md` first and then fix any supporting references in `README.md`, `docs/QUICK_REFERENCE.md`, and migration notes.

## Component Polymorphism

- mockgal's render pipeline dispatches polymorphically over a `Component` ABC (`mockgal.py` Section 2). Every concrete profile must implement four methods: `to_libprofit_spec(ctx)`, `to_astropy_image(ctx, shape)`, `derived_params(ctx)`, and `angular_extent_arcsec(ctx)`. Adding a new profile is then a single class plus one registry entry in `_COMPONENT_REGISTRY`, not three dispersed `isinstance` branches.
- `RenderContext` bundles per-render environment (`redshift`, `pixel_scale`, `zeropoint`, image-center xy) so components can convert their intrinsic kpc/abs_mag fields into image-plane parameters without holding a reference to `ImageConfig` or `MockGalaxy`.
- The libprofit backend (via `profit-cli`) renders all three current profile types natively (`sersic:`, `ferrer:`, `psf:`). The astropy backend renders Sersic and PointSource directly; `FerrerComponent.to_astropy_image` raises `NotImplementedError` because astropy.modeling has no native Ferrer profile, and a custom Fittable2DModel was deemed out of scope for a fallback path.
- `PointSourceComponent` requires `ImageConfig.psf_enabled=True`. `MockImageGenerator.generate()` enforces this at the top of the call rather than silently rendering a single bright pixel — that would be an isophote-fitting footgun.
- Image-size auto-sizing falls back to `MIN_IMAGE_EXTENT_PIX = 51` when every component has zero intrinsic extent (e.g. PSF-only galaxies). Without the fallback the `2*size_factor*overall_re_pix + 1` formula collapses to size 1.
- The polymorphic refactor was behavior-preserving: `max abs diff = 0.0` against a pre-refactor reference render of a two-Sersic galaxy. Always keep a stash of pre-refactor outputs around when restructuring a render pipeline.

## GALFIT-Backed Reference Renders

- `mockgal_galfit.py` wraps the GALFIT 3.0.7 binary (default `/Users/shuang/code/galfit/galfit`, overridable via `GALFIT_BIN`) in `P) 1` model-only mode. It supports two input modes: a `MockGalaxy` (translated through `mockgalaxy_to_galfit_dict`) or a galfit_io `from_dict`-shaped dict (used to re-render parsed Salo+2015 P4 outgals against their original `*_subcomps.fits`).
- For the same single-Sersic input, mockgal-libprofit/astropy and mockgal-galfit agree at correlation > 0.95 with flux ratio within 2%. Differences come from independent numerical quadratures and sub-pixel sampling, not bugs.
- Use `mockgal_galfit.py` as the canonical reference renderer when pixel-perfect parity matters; use the libprofit/astropy paths in mockgal for production scale.
- The bundled `/galfit` skill at `~/.claude/skills/galfit/scripts/galfit_io.py` provides `read_galfit`, `write_galfit`, `from_dict`, `to_dict`. mockgal_galfit imports it via `sys.path` insertion at module load — keep the skill installed.

## libprofit Operational Notes

- Built `profit-cli` binaries on this machine often have a hardcoded `@rpath` that resolves to a non-existent `/Users/mac/...` path. Set `DYLD_LIBRARY_PATH` to the directory containing `libprofit.dylib` before invoking the binary, or rebuild with the correct rpath.
- Treat profit-cli as available only if it can actually start. mockgal's `engine='auto'` selection runs a health check and falls back to astropy when the binary fails to load — preserve that behavior in any future refactor.
