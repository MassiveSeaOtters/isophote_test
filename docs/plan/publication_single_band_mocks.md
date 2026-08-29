# Publication single-band mock images

Date: 2026-08-29
Branch: `feature/publication-single-band-mocks`
Status: complete

## Goal

Generate new Huang2013 and CS4G/S4G single-band publication test images without
modifying the existing external datasets.

## Contract

- Output roots are new folders under
  `/Volumes/galaxy/isophote/publication_single_band_round1_2026_08_28/mock_data/`.
- Force the libprofit renderer and fail if MockGal falls back to another
  renderer.
- Include genuinely noiseless, HSC-wide, and HSC-deep scenarios.
- Noisy scenarios use a recorded campaign base seed and a stable SHA-256
  derivation over base seed, galaxy name, and scenario name.
- Record the realized seed in each FITS header and in `run_metadata.json`.
- Preserve historical manifest behavior by making per-galaxy seed derivation
  opt-in.
- Validate two galaxies from each dataset before broader generation.

## Verification

1. Unit-check stable and distinct seed derivation.
2. Unit-check fixed-seed backward compatibility.
3. Unit-check FITS and run-metadata provenance.
4. Dry-run both publication manifests.
5. Render two galaxies per dataset into new smoke folders and verify image
   counts, renderer, noiseless flags, and seed uniqueness.

## Review

- Focused tests: 91 passed and 5 skipped.
- Huang2013 smoke: 2 galaxies, 9 scenarios each, 18 FITS images.
- CS4G/S4G smoke: 2 galaxies, 6 scenarios each, 12 FITS images.
- Every smoke image is finite and reports `ENGINE=libprofit`.
- Noiseless rows contain no noise seed; all 24 noisy rows use distinct seeds
  that match `run_metadata.json`.
- Regenerating `ESO 185-G054` in `deep_z005` in a separate folder produced a
  byte-identical FITS file with SHA-256
  `95261a45d00811deea5d70dd1da029aea4d3685d4ce2f7c4c1e82e1ea2d31b30`.
- Smoke outputs are preflight artifacts. The full publication run must start
  from a clean merged commit so its provenance records `git_dirty=false`.
