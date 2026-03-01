# Lessons Learned

## Execution Safety

- Validate Huang2013 workflows on a small subset before attempting a broader batch run.
- On memory-limited machines, run Huang2013 jobs with `--workers 1`; parallel workers can push large images over memory limits.
- Keep the 4001-pixel image-size cap in place unless the user explicitly asks to revisit it.

## Huang2013 Data Conventions

- The Huang2013 `VMag` column is already an absolute V-band magnitude. Do not apply a distance modulus conversion during catalog conversion or mock generation.
- The default Huang2013 redshift assumption is `z = 0.01` for the source models unless a workflow intentionally overrides it for simulated observations.
- One Huang2013 galaxy has `n > 8`; downstream logic clamps extreme Sersic indices, so changes to parser or validation code should preserve that safety behavior.

## Noise And Sky Handling

- Sky should be added internally for Poisson statistics and subtracted before writing the final mock image. Leaving sky in the output breaks the intended sky-free mock convention.
- `sky_sb_value` and `sky_enabled` should remain decoupled. The sky value can be needed for noise modeling even when no sky background should survive in the saved image.
- Randomized noise seeds should be opt-in so science mocks can vary while regression tests remain reproducible.

## Image Size And Survey Mocking

- Dynamic sizing with `size_factor = 16` is preferable to hard-coded 4000-pixel Huang2013 images because it scales with angular size while still respecting the 4001-pixel cap.
- The current Huang2013 systematic mock workflow uses HSC-like settings: `pixel_scale = 0.168`, `psf_fwhm = 0.7`, and `sky_sb_limit = 24.5`.
- Per-galaxy QA mosaics are useful enough to keep in the canonical Huang2013 generator because they provide quick visual validation without opening four FITS files individually.

## Filename And Output Conventions

- Output filenames must use `sanitize_filename()` so galaxy-name spaces are removed instead of converted to underscores.
- Use a single underscore only between the sanitized galaxy name and the config name, for example `NGC3923_clean.fits`.
- Tests and benchmarks should write into organized `output/` subdirectories rather than cluttering the repo root.

## Documentation Maintenance

- Agent-facing files should reference lessons and workflows rather than duplicating long command catalogs or session history.
- Session-specific status, commit hashes, and handoff notes belong in `docs/journal/` or dedicated session logs, not in `AGENTS.md` or `CLAUDE.md`.
- When the canonical Huang2013 workflow changes, update `inputs/README.md` first and then fix any supporting references in `README.md`, `docs/QUICK_REFERENCE.md`, and migration notes.
