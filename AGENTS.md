# AGENTS.md

## Purpose

This repository generates mock galaxy images for isophote-fitting tests, with the Huang et al. (2013) sample as the primary systematic workflow.

## Source Of Truth

- Repo workflow rules: `AGENTS.md`
- Claude-oriented project summary: `CLAUDE.md`
- Durable lessons learned: `docs/LESSON.md`
- Current cleanup/task tracking: `docs/todo.md` and `docs/plan/`
- Canonical `inputs/` inventory and paths: `inputs/README.md`

## Working Rules

- Work on a feature branch, never directly on `main`.
- Keep all code, docs, commits, and logs in English.
- Use `snake_case` naming in Python and avoid camelCase.
- For non-trivial work, update `docs/todo.md` and the relevant plan file under `docs/plan/`.
- Agent-written code changes must include tests or an explicit explanation of why tests were not added.
- End each task or session with the proposed next step.

## Huang2013 Execution Safety

- Do not run the full 93-galaxy Huang2013 batch on memory-limited machines.
- Default to `--workers 1` for Huang2013 validation runs unless the user explicitly wants broader parallel testing.
- Respect the 4001-pixel image-size cap unless the user explicitly asks to change it.
- Use small-scale validation before any expensive batch run.

## Data And Output Conventions

- Huang2013 `VMag` values are already absolute magnitudes; do not apply a distance modulus conversion.
- Use `sanitize_filename()` for output names. Galaxy-name spaces are removed, not converted to underscores.
- Keep canonical Huang2013 assets under `inputs/huang2013/`; demos and examples belong in their own subdirectories.

## Documentation Policy

- Agent files stay short and durable.
- Operational pitfalls, rationale, and accumulated conventions belong in `docs/LESSON.md`.
- Session-specific history belongs in `docs/journal/` or dedicated session notes, not in agent files.
