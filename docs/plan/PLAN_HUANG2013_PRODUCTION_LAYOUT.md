# Huang2013 Production Layout Plan

## Goal

Reorganize the Huang2013 workflow so HSC `i`-band calibration results are documented as Huang2013-specific references, while the production-oriented mock-generation path stays explicit and does not turn those measurements into global defaults.

## Scope

1. Keep the measured HSC `i`-band depth results in a dedicated Huang2013 reference file.
2. Replace the hardcoded Huang2013 batch bundle with manifest-driven production runs that copy numeric depth values inline.
3. Document a cleaner repo layout and command split between:
   - libprofit-backed Huang2013 validation
   - production-oriented systematic mock generation
4. Add focused tests for manifest parsing, validation, and run-artifact writing.

## Constraints

- HSC calibration values currently apply only to the `i` band.
- The measured `wide` and `dud` values should stay scoped to Huang2013 workflow references.
- The production workflow should be reproducible from explicit manifest rows rather than runtime profile selection.
- `LIBPROFIT_PATH` must be re-verified before relying on libprofit-backed Huang2013 validation commands in this shell.

## Planned Outputs

- `inputs/huang2013/configs/huang2013_hsc_i_calibration.yaml`
- `inputs/huang2013/README.md`
- `inputs/huang2013/runs/`
- updated `inputs/huang2013/scripts/generate_huang2013_mocks.py`
- aligned references in `inputs/README.md`, `README.md`, `docs/QUICK_REFERENCE.md`, `docs/LESSON.md`, and `docs/todo.md`
- focused generator tests

## Validation

- Add tests for manifest loading, validation, config selection, and artifact writing.
- Run focused pytest targets plus a small manifest-backed smoke run; do not start a full Huang2013 production batch as part of this reorganization.
