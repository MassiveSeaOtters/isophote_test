#!/usr/bin/env python
"""
galfit_constraints.py - Parser and writer for GALFIT constraint files.

Constraint files are the companion to the ``G)`` header entry of a GALFIT
config. Grammar (six forms, one entry per line):

    C         param   low high          # soft, relative to initial value
    C         param   low to high       # soft, absolute range (keyword 'to')
    C1_C2_..  param   offset            # hard coupling, preserve offsets
    C1_C2_..  param   ratio             # hard coupling, preserve ratios
    CA-CB     param   low high          # soft pairwise difference
    CA/CB     param   low high          # soft pairwise ratio

The ``param`` column accepts the documented name table:
    x, y, mag, re (or rs -- equivalent), n, alpha, beta, gamma, pa, q, c,
    f<N>a, f<N>p, b<N>, r<N>, ...
or the classical integer parameter number (1-10 plus the hidden-block
numeric identifier for each mode).

Public surface:

    entries = read_constraints("constraints.txt")
    write_constraints(entries, "constraints.txt")
    to_yaml_constraints(entries) / from_yaml_constraints(text)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# -----------------------------------------------------------------------------
# Known parameter-name vocabulary (for validation/autocomplete).
# -----------------------------------------------------------------------------

CLASSICAL_NAMES = ("x", "y", "mag", "re", "rs", "n", "alpha", "beta", "gamma",
                   "pa", "q", "c")
MODE_NAME_RE = re.compile(r"^(f\d+[ap]|b\d+|r\d+)$", re.IGNORECASE)


def is_known_param_name(name: str) -> bool:
    """Return True if ``name`` is in the documented parameter-name table
    or looks like a legitimate hidden-block mode name."""
    low = name.lower()
    if low in CLASSICAL_NAMES:
        return True
    if MODE_NAME_RE.match(low):
        return True
    if low.isdigit():
        return True
    return False


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

CouplingKind = Literal[
    "single",        # one component, soft range (relative or absolute)
    "offset",        # C1_C2_..., hard offset lock
    "ratio",         # C1_C2_..., hard ratio lock
    "diff",          # CA-CB, soft pairwise difference
    "ratio_bounds",  # CA/CB, soft pairwise ratio range
]


@dataclass
class ConstraintEntry:
    """
    One line in a GALFIT constraint file.

    - For ``coupling == "single"`` and ``"diff"`` and ``"ratio_bounds"``:
      ``bounds`` is a ``(low, high)`` tuple. ``absolute`` is True when the
      soft range was written as ``low to high`` and False when it was
      written as ``low high`` (i.e. relative to the input value).
    - For ``coupling == "offset"`` and ``"ratio"``: ``bounds`` is the
      literal keyword (``"offset"`` or ``"ratio"``) and ``absolute`` is
      unused.
    """
    components: list[int]
    parameter: str
    coupling: CouplingKind = "single"
    bounds: Union[tuple[float, float], str] = ("nan", "nan")  # type: ignore[assignment]
    absolute: bool = False
    comment: Optional[str] = None


# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------

def _strip_comment(line: str) -> tuple[str, Optional[str]]:
    if "#" not in line:
        return line.rstrip(), None
    body, _, comment = line.partition("#")
    return body.rstrip(), comment.strip()


def _parse_components(col: str) -> tuple[list[int], CouplingKind]:
    """Parse the first column into (component list, coupling kind).

    Separator semantics:
        ``_``   -> hard coupling of >=2 components (kind narrows to
                   offset/ratio by the range column, decided later).
        ``-``   -> soft pairwise difference of exactly 2 components.
        ``/``   -> soft pairwise ratio bounds of exactly 2 components.
        none    -> single component; kind is "single".
    """
    if "_" in col:
        parts = col.split("_")
        return [int(p) for p in parts], "offset"  # may be reclassified to "ratio"
    if "-" in col:
        parts = col.split("-")
        if len(parts) != 2:
            raise ValueError(f"diff-separator expects exactly 2 components: {col!r}")
        return [int(parts[0]), int(parts[1])], "diff"
    if "/" in col:
        parts = col.split("/")
        if len(parts) != 2:
            raise ValueError(f"ratio-separator expects exactly 2 components: {col!r}")
        return [int(parts[0]), int(parts[1])], "ratio_bounds"
    return [int(col)], "single"


def _parse_range(tokens: list[str]) -> tuple[Union[tuple[float, float], str], bool]:
    """Parse the range column tokens and return (bounds, absolute)."""
    if len(tokens) == 1:
        word = tokens[0].lower()
        if word in ("offset", "ratio"):
            return word, False
        raise ValueError(f"expected 'offset' or 'ratio', got {tokens[0]!r}")
    if len(tokens) == 3 and tokens[1].lower() == "to":
        return (float(tokens[0]), float(tokens[2])), True
    if len(tokens) == 2:
        return (float(tokens[0]), float(tokens[1])), False
    raise ValueError(f"cannot parse range column: {tokens!r}")


# -----------------------------------------------------------------------------
# Reader
# -----------------------------------------------------------------------------

def read_constraints_text(text: str) -> list[ConstraintEntry]:
    entries: list[ConstraintEntry] = []
    for raw in text.splitlines():
        body, comment = _strip_comment(raw)
        if not body.strip():
            continue
        tokens = body.split()
        if len(tokens) < 3:
            raise ValueError(f"constraint line too short: {raw!r}")
        components, coupling = _parse_components(tokens[0])
        parameter = tokens[1]
        if not is_known_param_name(parameter):
            # Accept but record the unfamiliar name; GALFIT itself may
            # still recognize it (the README is non-exhaustive).
            pass
        bounds, absolute = _parse_range(tokens[2:])
        # Reclassify hard-coupling entries based on the range keyword.
        if coupling == "offset" and bounds == "ratio":
            coupling = "ratio"
        entries.append(ConstraintEntry(
            components=components,
            parameter=parameter,
            coupling=coupling,
            bounds=bounds,
            absolute=absolute,
            comment=comment,
        ))
    return entries


def read_constraints(path: Union[str, Path]) -> list[ConstraintEntry]:
    return read_constraints_text(Path(path).read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------------

def _format_component_column(entry: ConstraintEntry) -> str:
    if entry.coupling in ("offset", "ratio"):
        return "_".join(str(c) for c in entry.components)
    if entry.coupling == "diff":
        return f"{entry.components[0]}-{entry.components[1]}"
    if entry.coupling == "ratio_bounds":
        return f"{entry.components[0]}/{entry.components[1]}"
    return str(entry.components[0])


def _format_range_column(entry: ConstraintEntry) -> str:
    if isinstance(entry.bounds, str):
        return entry.bounds
    low, high = entry.bounds
    low_s = _format_number(low)
    high_s = _format_number(high)
    if entry.absolute:
        return f"{low_s} to {high_s}"
    return f"{low_s} {high_s}"


def _format_number(x: float) -> str:
    if x == int(x) and abs(x) < 1e16:
        return f"{x:.1f}"
    return repr(x)


def write_constraints_text(entries: list[ConstraintEntry]) -> str:
    lines: list[str] = []
    lines.append("# Component/    parameter   constraint")
    lines.append("# operation     (see below) range")
    lines.append("")
    for e in entries:
        comp_col = _format_component_column(e)
        range_col = _format_range_column(e)
        line = f"  {comp_col:<12} {e.parameter:<8} {range_col}"
        if e.comment:
            line += f"    # {e.comment}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def write_constraints(
    entries: list[ConstraintEntry], path: Union[str, Path],
) -> None:
    Path(path).write_text(write_constraints_text(entries), encoding="utf-8")


# -----------------------------------------------------------------------------
# Dict / YAML / JSON
# -----------------------------------------------------------------------------

def entries_to_dicts(entries: list[ConstraintEntry]) -> list[dict]:
    out: list[dict] = []
    for e in entries:
        d: dict[str, Any] = {
            "components": list(e.components),
            "parameter": e.parameter,
            "coupling": e.coupling,
        }
        if isinstance(e.bounds, tuple):
            d["bounds"] = list(e.bounds)
            d["absolute"] = e.absolute
        else:
            d["bounds"] = e.bounds
        if e.comment:
            d["comment"] = e.comment
        out.append(d)
    return out


def entries_from_dicts(data: list[dict]) -> list[ConstraintEntry]:
    entries: list[ConstraintEntry] = []
    for d in data:
        bounds_raw = d["bounds"]
        if isinstance(bounds_raw, list):
            bounds: Union[tuple[float, float], str] = (
                float(bounds_raw[0]), float(bounds_raw[1]),
            )
        else:
            bounds = str(bounds_raw)
        entries.append(ConstraintEntry(
            components=[int(c) for c in d["components"]],
            parameter=d["parameter"],
            coupling=d["coupling"],
            bounds=bounds,
            absolute=bool(d.get("absolute", False)),
            comment=d.get("comment"),
        ))
    return entries


def to_yaml_constraints(entries: list[ConstraintEntry]) -> str:
    import yaml as _yaml
    return _yaml.safe_dump(entries_to_dicts(entries), sort_keys=False)


def from_yaml_constraints(text: str) -> list[ConstraintEntry]:
    import yaml as _yaml
    return entries_from_dicts(_yaml.safe_load(text))


def to_json_constraints(
    entries: list[ConstraintEntry], *, indent: int = 2,
) -> str:
    return json.dumps(entries_to_dicts(entries), indent=indent)


def from_json_constraints(text: str) -> list[ConstraintEntry]:
    return entries_from_dicts(json.loads(text))
