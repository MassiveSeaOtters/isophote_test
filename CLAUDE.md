# CLAUDE.md

## Project

MockGal generates mock galaxy images for isophote-fitting tests using either the `libprofit` backend or the pure-Python `astropy` backend. The Huang et al. (2013) sample under `inputs/huang2013/` is the main systematic test dataset.

## Code Map

- `mockgal.py`: CLI entry point and core library
- `inputs/huang2013/`: canonical Huang2013 catalog, models, configs, and workflow scripts
- `inputs/examples/`: small reusable example models and configs
- `inputs/demos/`: one-off demo scripts
- `tests/test_mockgal.py`: test suite
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
