# CLAUDE.md

## Project

MockGal generates mock galaxy images for isophote-fitting tests. It renders multi-component galaxies (Sersic + Ferrer bar + PSF-convolved nucleus) via a polymorphic `Component` ABC, with three rendering backends: `libprofit` (all profile types), `astropy` (Sersic + PointSource only), and the GALFIT binary via `mockgal_galfit.py`. The Huang et al. (2013) sample under `inputs/huang2013/` is the main systematic test dataset.

## Code Map

- `mockgal.py`: CLI entry point and core library — `Component` ABC, `SersicComponent`, `FerrerComponent`, `PointSourceComponent`, `RenderEngine` (libprofit + astropy paths), `MockImageGenerator`
- `mockgal_galfit.py`: GALFIT-backed reference renderer (wraps the GALFIT binary in `P) 1` model-only mode); two input modes — MockGalaxy + GALFIT-dict-direct
- `inputs/huang2013/`: canonical Huang2013 catalog, models, configs, and workflow scripts
- `inputs/examples/`: small reusable example models and configs
- `inputs/demos/`: one-off demo scripts
- `tests/test_mockgal.py`: core suite + Component-ABC contract test
- `tests/test_ferrer_component.py`, `tests/test_pointsource_component.py`: per-profile tests
- `tests/test_mockgal_galfit.py`: GALFIT-backed renderer tests (auto-skipped if binary missing at `$GALFIT_BIN`)
- `benchmarks/bench_engines.py`: benchmark runner

## Use These Docs

- `AGENTS.md`: repo workflow rules for agents
- `docs/LESSON.md`: durable lessons and pitfalls
- `inputs/README.md`: canonical `inputs/` table of contents and Huang2013 workflow
- `docs/PROFIT_CLI_USAGE.md`: raw `profit-cli` help text
- `docs/LIBPROFIT_COMPILE.md`: local build notes for `libprofit`

## Notes

- Keep agent-facing files short; operational detail belongs in `docs/LESSON.md`.
- Prefer canonical paths under `inputs/huang2013/`, `inputs/examples/`, and `inputs/demos/`.
- Use small Huang2013 validation runs before any expensive batch generation.
- New `Component` profile types must implement the four abstract methods and register in `mockgal._COMPONENT_REGISTRY`; the contract test in `tests/test_mockgal.py` will fail otherwise.
- `mockgal_galfit.py` and `tests/test_mockgal_galfit.py` rely on a working GALFIT binary. Set `GALFIT_BIN` env var to override the default `/Users/shuang/code/galfit/galfit`.
