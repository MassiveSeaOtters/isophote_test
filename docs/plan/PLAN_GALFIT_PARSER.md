# PLAN: GALFIT Config/Output File Parser

## Goal

Add a comprehensive, round-trippable parser for GALFIT's structured-ASCII
configuration files (`.galfit`, `.feedme`, `galfit.NN` restart files) and its
companion constraint files. Target GALFIT 3.0.4+ (binary on this machine is at
`/Users/shuang/code/galfit/galfit`). Provide YAML and JSON serialization so the
same model can travel through the wider MockGal workflow.

## Scope

### Must cover

- Header keys `A)`–`P)` in any order, including the multi-token forms:
  `D) psf kernel`, `H) xmin xmax ymin ymax`, `I) nx ny`, `K) dx dy`. Accept
  `P) 0|1|2|3` (3 = write `subcomps.fits`).
- All 11 profile types with profile-specific parameter semantics:
  `sersic`, `devauc`, `expdisk`, `edgedisk`, `gaussian`, `moffat`, `nuker`,
  `king`, `ferrer`, `psf`, `sky`. Plus the `<name>1|2|3` flux-normalization
  suffix (`sersic2`, `gaussian1`, etc.).
- All hidden blocks that attach to the preceding component:
  - `Z)` skip flag.
  - `C0)` diskyness/boxyness.
  - `Bn)` bending modes (sparse positive integer index, single value + fit).
  - `Fn)` Fourier modes (sparse index, two values + two fit flags).
  - `R0)..R10)` coordinate rotation (R0 is an enum string:
    `powerlaw | log | none`; accept `tanh | sqrt | linear` with a warning).
  - Truncation pseudo-component `T0)..T10)` with `T0` enum string
    `radial | radial-b | radial2 | radial2-b | length | height`.
  - Linking fields `Ti) N` and `To) N` on light components (single integer,
    no fit flag).
- Output-file-only decorations that a clean input file does not carry:
  - `(err)` per-value uncertainties appended on the same line.
  - `[value]` for input fit-flag-0 (held fixed).
  - `{value}` for values pinned by an active constraint.
  - `*value*` for numerically suspicious values (3.0.1+).
  - Decoration composition is handled in any order.
- Constraint files with all six line grammars from `EXAMPLE.CONSTRAINTS`:
  - `C param low high` (soft, relative to input).
  - `C param low to high` (soft, absolute range).
  - `C1_C2_..._Ck param offset` (hard offset coupling).
  - `C1_C2_..._Ck param ratio` (hard ratio coupling).
  - `CA-CB param low high` (soft pairwise difference).
  - `CA/CB param low high` (soft pairwise ratio).
  - Parameter column accepts the documented name table (`x`, `y`, `mag`,
    `re | rs`, `n`, `alpha`, `beta`, `gamma`, `pa`, `q`, `c`, `f<N>a`,
    `f<N>p`, `b<N>`, `r<N>`) or the classical integer parameter number.
- Round-trip: parse, serialize to YAML or JSON, load back, write `.galfit`
  that is semantically identical to the original.

### Deliberately out of scope

- Automatic upgrade of pre-3.0 files (parameter 8 is axis ratio). Warn only.
- Filenames containing whitespace (not supported by GALFIT itself).
- Preserving exact whitespace and column alignment of the original file. The
  parser preserves semantics, not typography.

## Approved Design

### Modules

- `galfit_io.py`: header, component dataclasses, `GalfitFile`, tokenizer,
  envelope parser, read/write, YAML/JSON.
- `galfit_constraints.py`: constraint-entry dataclass, parser, writer.
- `tests/test_galfit_io.py`, `tests/test_galfit_constraints.py`.
- `tests/fixtures/galfit/`: one fixture per profile, per hidden-block case,
  per constraint grammar, plus a composite output-decoration fixture and
  one fixture for a real-GALFIT round-trip integration test.

### Data model (tagged union)

- `FittedValue`: `value: float`, `fit: bool`, `uncertainty: float | None`,
  `fixed_in_output: bool`, `constrained_in_output: bool`, `suspicious: bool`.
- `HiddenBlocks`: `z_skip`, `c0`, `bending: dict[int, FittedValue]`,
  `fourier: dict[int, (FittedValue, FittedValue)]`, `rotation: RotationBlock | None`,
  `trunc_inner_ref: int | None`, `trunc_outer_ref: int | None`.
- One `*Component` dataclass per profile type, each declaring its own
  profile-specific parameter fields (`SersicComponent.re`, `NukerComponent.mu_rb`,
  etc.). All share the common `x`, `y`, `hidden` fields and a `profile` literal.
- `TruncationComponent`: a standalone pseudo-component corresponding to the
  `T0)` block with a `trunc_type` enum.
- `GalfitHeader`: all 15 header keys with typed fields.
- `GalfitFile`: `header`, `components: list[Component]`.
- `ConstraintEntry`: `components: list[int]`, `coupling: Literal[...]`,
  `parameter: str`, `bounds: ...`, `absolute: bool`.

### Round-trip posture

When writing an output file (`galfit.NN`) back to disk, default to stripping
output-only decorations (`[]`, `{}`, `*`, `(err)`) and emitting a clean input
file. Opt-in `preserve_decorations=True` keeps them for diagnostic dumps.

### Validation posture

- Reject structurally broken lines (missing profile name, mismatched column
  count for a known-shape line like `1)`).
- Warn but accept unknown `R0)` / `T0)` strings.
- Warn but accept pre-3.0 `8)` usage (axis ratio in old numbering).
- Warn when a value-envelope composition is unexpected (e.g. `*[value]*`).

## Phase Plan

- Phase 2a: data model + tokenizer + envelope parser + basic tests.
- Phase 2b: header + profile readers + writers + round-trip tests.
- Phase 2c: hidden blocks reading/writing + tests.
- Phase 2d: constraint-file reader/writer + tests.
- Phase 2e: YAML and JSON serialization + round-trip tests.
- Phase 2f: integration test that runs `/Users/shuang/code/galfit/galfit`
  against a synthesized input and verifies parser round-trip on its
  `galfit.01` output.
- Phase 3: SKILL package for global install at `~/.claude/skills/galfit/`.

## Acceptance Criteria

- Every fixture round-trips byte-for-byte after `parse -> serialize -> parse`
  idempotence, and semantically after `parse -> write -> parse`.
- Running GALFIT against a parser-written input produces a `galfit.01` that
  the parser can read back without warnings.
- All 11 profiles plus all hidden blocks plus all 6 constraint grammars are
  covered by tests.
