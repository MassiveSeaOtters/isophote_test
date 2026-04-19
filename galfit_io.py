#!/usr/bin/env python
"""
galfit_io.py - Parser and writer for GALFIT configuration and output files.

GALFIT (Peng et al. 2002, 2010) stores its input configuration and its
post-fit restart/output files in the same structured-ASCII format. This
module provides a round-trippable parser for that format (GALFIT 3.0.4+),
plus YAML and JSON serialization.

Top-level surface:

    file = read_galfit("galfit.feedme")          # -> GalfitFile
    write_galfit(file, "next.feedme")            # strips output decorations
    write_galfit(file, "dump.feedme", preserve_decorations=True)

    yaml_text = to_yaml(file)
    file2 = from_yaml(yaml_text)
    json_text = to_json(file)
    file3 = from_json(json_text)

Constraint files live in the companion module `galfit_constraints.py`.

Supported profiles (with optional flux-normalization suffix 1|2|3):
    sersic, devauc, expdisk, edgedisk, gaussian, moffat, nuker, king,
    ferrer, psf, sky.

Supported hidden blocks: Z, C0, B<N>, F<N>, R0..R10, T0..T10, Ti, To.

Output-file value decorations are captured on the FittedValue and stripped
by default on write:
    bare value        -> free fitted
    (err)             -> 1-sigma uncertainty on the same line
    [value]           -> held fixed via a fit_flag=0 in input
    {value}           -> pinned by an active constraint
    *value*           -> numerically suspicious (GALFIT >= 3.0.1)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional, Sequence, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: Constants
# =============================================================================

VALID_PROFILES = (
    "sersic", "devauc", "expdisk", "edgedisk", "gaussian", "moffat",
    "nuker", "king", "ferrer", "psf", "sky",
)

VALID_FLUX_NORM_SUFFIXES = (1, 2, 3)

# R0) enum strings. README documents only powerlaw | log | none; the
# EXAMPLE.INPUT comment mentions tanh | sqrt | linear as aspirational -
# accept them with a warning.
ROTATION_TYPES_DOCUMENTED = ("powerlaw", "log", "none")
ROTATION_TYPES_ASPIRATIONAL = ("tanh", "sqrt", "linear", "power")

# T0) enum strings. length|height are edgedisk-only per README Section 6.2.
TRUNCATION_TYPES = (
    "radial", "radial-b", "radial2", "radial2-b", "length", "height",
)

# Header key letter -> (field name on GalfitHeader, tokenizer kind).
# Tokenizer kinds:
#   "str1"   single string token
#   "int1"   single integer
#   "flt1"   single float
#   "int4"   four integers (fit region)
#   "int2"   two integers (convolution box)
#   "flt2"   two floats (plate scale)
#   "psf"    one to two string tokens (psf file [diffusion kernel])
HEADER_SCHEMA = {
    "A": ("input_image",        "str1"),
    "B": ("output_block",       "str1"),
    "C": ("sigma_image",        "str1"),
    "D": ("psf_image",          "psf"),
    "E": ("psf_fine_sampling",  "int1"),
    "F": ("bad_pixel_mask",     "str1"),
    "G": ("constraint_file",    "str1"),
    "H": ("fit_region",         "int4"),
    "I": ("conv_box",           "int2"),
    "J": ("zeropoint",          "flt1"),
    "K": ("plate_scale",        "flt2"),
    "O": ("display_type",       "str1"),
    "P": ("options",            "int1"),
}


# =============================================================================
# Section 2: FittedValue - value + fit flag + output decorations
# =============================================================================

@dataclass
class FittedValue:
    """
    A single GALFIT parameter value with its fit flag and any output-file
    decorations.

    The three decoration envelopes are mutually independent and can compose;
    README Section 10 states brackets/braces are outermost and stars are
    innermost, but the parser tolerates any nesting order.
    """
    value: float
    fit: bool = True
    uncertainty: Optional[float] = None
    fixed_in_output: bool = False         # [value] in output
    constrained_in_output: bool = False   # {value} in output
    suspicious: bool = False              # *value* in output

    def to_token(self, *, preserve_decorations: bool = False) -> str:
        """Render as a single whitespace-free token."""
        s = _format_number(self.value)
        if not preserve_decorations:
            return s
        if self.suspicious:
            s = f"*{s}*"
        if self.fixed_in_output:
            s = f"[{s}]"
        elif self.constrained_in_output:
            s = f"{{{s}}}"
        return s

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"value": self.value, "fit": self.fit}
        if self.uncertainty is not None:
            d["uncertainty"] = self.uncertainty
        for flag in ("fixed_in_output", "constrained_in_output", "suspicious"):
            if getattr(self, flag):
                d[flag] = True
        return d

    @classmethod
    def from_dict(cls, data: Union[dict, float, int]) -> "FittedValue":
        if isinstance(data, (int, float)):
            return cls(value=float(data))
        unc_raw = data.get("uncertainty")
        return cls(
            value=float(data["value"]),
            fit=bool(data.get("fit", True)),
            uncertainty=float(unc_raw) if unc_raw is not None else None,
            fixed_in_output=bool(data.get("fixed_in_output", False)),
            constrained_in_output=bool(data.get("constrained_in_output", False)),
            suspicious=bool(data.get("suspicious", False)),
        )


def _format_number(x: float) -> str:
    """Compact, round-trip-safe float rendering. Integers get a single
    decimal place; everything else uses ``repr`` so Python's shortest
    round-trippable representation is preserved (PEP 3101 behavior)."""
    if x == int(x) and abs(x) < 1e16:
        return f"{x:.1f}"
    return repr(x)


# Envelope regex: accepts nested order of [], {}, *  around a number. We
# strip layer by layer so the order is not significant.
_NUMBER_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _strip_envelope(token: str) -> tuple[str, bool, bool, bool]:
    """Peel off [...] / {...} / *...* wrappers. Returns (inner, fixed,
    constrained, suspicious). Any order of nesting is accepted."""
    fixed = constrained = suspicious = False
    changed = True
    while changed:
        changed = False
        if len(token) >= 2 and token[0] == "[" and token[-1] == "]":
            token = token[1:-1]
            fixed = True
            changed = True
        elif len(token) >= 2 and token[0] == "{" and token[-1] == "}":
            token = token[1:-1]
            constrained = True
            changed = True
        elif len(token) >= 2 and token[0] == "*" and token[-1] == "*":
            token = token[1:-1]
            suspicious = True
            changed = True
    return token, fixed, constrained, suspicious


def _parse_number(token: str) -> float:
    """Parse a GALFIT numeric token after envelope stripping."""
    m = _NUMBER_RE.match(token)
    if not m:
        raise ValueError(f"cannot parse numeric token: {token!r}")
    return float(token)


# =============================================================================
# Section 3: Hidden blocks (Z, C0, B*, F*, R*, T-ref)
# =============================================================================

@dataclass
class RotationBlock:
    """Coordinate rotation block (R0..R10) attached to a light component."""
    mode: str = "none"                            # R0)
    r_in: Optional[FittedValue] = None            # R1) bar radius
    r_out: Optional[FittedValue] = None           # R2) outer radius
    theta_out: Optional[FittedValue] = None       # R3) cumulative rotation
    alpha_or_rws: Optional[FittedValue] = None    # R4) powerlaw alpha or log winding
    incl: Optional[FittedValue] = None            # R9) inclination
    sky_pa: Optional[FittedValue] = None          # R10) sky PA
    # Slots R5..R8 are undocumented but legal per constraint grammar; keep
    # them sparse so the occasional non-standard file round-trips.
    extra: dict[int, FittedValue] = field(default_factory=dict)


@dataclass
class HiddenBlocks:
    """Attached hidden blocks; all fields optional."""
    z_skip: int = 0                                     # Z)
    c0: Optional[FittedValue] = None                    # C0)
    bending: dict[int, FittedValue] = field(default_factory=dict)    # B<N>
    fourier: dict[int, tuple[FittedValue, FittedValue]] = \
        field(default_factory=dict)                                   # F<N>
    rotation: Optional[RotationBlock] = None            # R0..R10
    trunc_inner_ref: Optional[int] = None               # Ti) N
    trunc_outer_ref: Optional[int] = None               # To) N


# =============================================================================
# Section 4: Component dataclasses (tagged union)
# =============================================================================

@dataclass
class _ComponentBase:
    """Shared state across all profile components."""
    x: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    y: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    flux_norm_suffix: Optional[int] = None
    hidden: HiddenBlocks = field(default_factory=HiddenBlocks)
    # GALFIT's restart files emit every numeric slot 1..10 for each light
    # profile, filling unused ones with zero-valued placeholders (e.g.
    # sersic output contains ``6) 7) 8)`` with fit_flag=0 and comment
    # "------"). Anything not in PARAM_SCHEMA for this profile goes here.
    extra_params: dict[int, FittedValue] = field(default_factory=dict)

    # Subclasses override.
    profile: ClassVar[str]
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]]   # num -> (attr, label)

    @property
    def profile_token(self) -> str:
        if self.flux_norm_suffix is None:
            return self.profile
        return f"{self.profile}{self.flux_norm_suffix}"


@dataclass
class SersicComponent(_ComponentBase):
    mag: FittedValue = field(default_factory=lambda: FittedValue(20.0))
    re: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    n: FittedValue = field(default_factory=lambda: FittedValue(4.0))
    q: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "sersic"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mag", "total magnitude"),
        4: ("re",  "R_e (effective radius) [pixels]"),
        5: ("n",   "Sersic index n (deVauc=4, expdisk=1)"),
        9: ("q",   "axis ratio (b/a)"),
        10: ("pa", "position angle (PA) [degrees: up=0, left=90]"),
    }


@dataclass
class DevaucComponent(_ComponentBase):
    mag: FittedValue = field(default_factory=lambda: FittedValue(20.0))
    re: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    q: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "devauc"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mag", "total magnitude"),
        4: ("re",  "R_e (effective radius) [pixels]"),
        9: ("q",   "axis ratio (b/a)"),
        10: ("pa", "position angle (PA) [degrees]"),
    }


@dataclass
class ExpdiskComponent(_ComponentBase):
    mag: FittedValue = field(default_factory=lambda: FittedValue(20.0))
    rs: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    q: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "expdisk"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mag", "total magnitude"),
        4: ("rs",  "R_s (disk scale length) [pixels]"),
        9: ("q",   "axis ratio (b/a)"),
        10: ("pa", "position angle (PA) [degrees]"),
    }


@dataclass
class EdgediskComponent(_ComponentBase):
    mu0: FittedValue = field(default_factory=lambda: FittedValue(20.0))
    h_s: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    r_s: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "edgedisk"
    # Edgedisk has no (9) axis ratio - geometry fixed by h_s / r_s.
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mu0", "central surface brightness [mag/arcsec^2]"),
        4: ("h_s", "disk scale-height [pixels]"),
        5: ("r_s", "disk scale-length [pixels]"),
        10: ("pa", "position angle (PA) [degrees]"),
    }


@dataclass
class GaussianComponent(_ComponentBase):
    mag: FittedValue = field(default_factory=lambda: FittedValue(20.0))
    fwhm: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    q: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "gaussian"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mag",  "total magnitude"),
        4: ("fwhm", "FWHM [pixels]"),
        9: ("q",    "axis ratio (b/a)"),
        10: ("pa",  "position angle (PA) [degrees]"),
    }


@dataclass
class MoffatComponent(_ComponentBase):
    mag: FittedValue = field(default_factory=lambda: FittedValue(20.0))
    fwhm: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    n: FittedValue = field(default_factory=lambda: FittedValue(1.5))
    q: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "moffat"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mag",  "total magnitude"),
        4: ("fwhm", "FWHM [pixels]"),
        5: ("n",    "powerlaw concentration index"),
        9: ("q",    "axis ratio (b/a)"),
        10: ("pa",  "position angle (PA) [degrees]"),
    }


@dataclass
class NukerComponent(_ComponentBase):
    mu_rb: FittedValue = field(default_factory=lambda: FittedValue(18.0))
    rb: FittedValue = field(default_factory=lambda: FittedValue(10.0))
    alpha: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    beta: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    gamma: FittedValue = field(default_factory=lambda: FittedValue(0.5))
    q: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "nuker"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mu_rb", "mu(R_b) [surface brightness mag/arcsec^2 at R_b]"),
        4: ("rb",    "R_b [pixels]"),
        5: ("alpha", "alpha (transition sharpness)"),
        6: ("beta",  "beta (outer powerlaw slope)"),
        7: ("gamma", "gamma (inner powerlaw slope)"),
        9: ("q",     "axis ratio (b/a)"),
        10: ("pa",   "position angle (PA) [degrees]"),
    }


@dataclass
class KingComponent(_ComponentBase):
    mu0: FittedValue = field(default_factory=lambda: FittedValue(15.0))
    rc: FittedValue = field(default_factory=lambda: FittedValue(10.0))
    rt: FittedValue = field(default_factory=lambda: FittedValue(50.0))
    alpha: FittedValue = field(default_factory=lambda: FittedValue(2.0))
    q: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "king"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mu0",   "mu(0) [central surface brightness, mag/arcsec^2]"),
        4: ("rc",    "R_c (core radius) [pixels]"),
        5: ("rt",    "R_t (truncation radius) [pixels]"),
        6: ("alpha", "alpha"),
        9: ("q",     "axis ratio (b/a)"),
        10: ("pa",   "position angle (PA) [degrees]"),
    }


@dataclass
class FerrerComponent(_ComponentBase):
    mu_central: FittedValue = field(default_factory=lambda: FittedValue(15.0))
    r_out: FittedValue = field(default_factory=lambda: FittedValue(50.0))
    alpha: FittedValue = field(default_factory=lambda: FittedValue(4.0))
    beta: FittedValue = field(default_factory=lambda: FittedValue(2.0))
    q: FittedValue = field(default_factory=lambda: FittedValue(1.0))
    pa: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "ferrer"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mu_central", "central surface brightness [mag/arcsec^2]"),
        4: ("r_out",      "outer truncation radius [pixels]"),
        5: ("alpha",      "alpha (outer truncation sharpness)"),
        6: ("beta",       "beta (central slope)"),
        9: ("q",          "axis ratio (b/a)"),
        10: ("pa",        "position angle (PA) [degrees]"),
    }


@dataclass
class PsfComponent(_ComponentBase):
    mag: FittedValue = field(default_factory=lambda: FittedValue(20.0))
    profile: ClassVar[str] = "psf"
    # Only params 1 (x,y) and 3 (mag); no size or shape.
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        3: ("mag", "total magnitude"),
    }


@dataclass
class SkyComponent(_ComponentBase):
    """Sky is special: parameters 1/2/3 are DC level / dsky/dx / dsky/dy,
    and there is no (x, y) coordinate. We keep the inherited x, y fields
    for uniform typing but ignore them; _sky_ignore_xy marks the override."""
    dc: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    dsky_dx: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    dsky_dy: FittedValue = field(default_factory=lambda: FittedValue(0.0))
    profile: ClassVar[str] = "sky"
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        1: ("dc",       "sky background DC [ADU counts]"),
        2: ("dsky_dx",  "dsky/dx (sky gradient in x)"),
        3: ("dsky_dy",  "dsky/dy (sky gradient in y)"),
    }


@dataclass
class TruncationComponent(_ComponentBase):
    """T0) pseudo-component. Produced in the file as a block starting with
    T0) <type>. May have attached F<N>/B<N> modes shaping the truncation."""
    trunc_type: str = "radial"                                       # T0)
    # T1) is the centroid (two-value with two fit flags), stored as x, y
    # on the base class. T2..T10 are profile-specific; we expose the
    # common ones explicitly and keep any stragglers in extra_params.
    t2: Optional[FittedValue] = None         # break radius (Type 2)
    t3: Optional[FittedValue] = None         # softening length or radius
    t4: Optional[FittedValue] = None         # sometimes break radius (Type 1)
    t5: Optional[FittedValue] = None         # sometimes softening length
    t9: Optional[FittedValue] = None         # optional axis ratio
    t10: Optional[FittedValue] = None        # optional PA
    extra_params: dict[int, FittedValue] = field(default_factory=dict)
    profile: ClassVar[str] = "__truncation__"  # not a real profile name
    PARAM_SCHEMA: ClassVar[dict[int, tuple[str, str]]] = {
        2: ("t2", "T2"),
        3: ("t3", "T3 (softening)"),
        4: ("t4", "T4 (break radius, Type 1)"),
        5: ("t5", "T5 (softening length, Type 1)"),
        9: ("t9", "axis ratio (optional)"),
        10: ("t10", "position angle (optional)"),
    }


PROFILE_REGISTRY: dict[str, type[_ComponentBase]] = {
    "sersic":   SersicComponent,
    "devauc":   DevaucComponent,
    "expdisk":  ExpdiskComponent,
    "edgedisk": EdgediskComponent,
    "gaussian": GaussianComponent,
    "moffat":   MoffatComponent,
    "nuker":    NukerComponent,
    "king":     KingComponent,
    "ferrer":   FerrerComponent,
    "psf":      PsfComponent,
    "sky":      SkyComponent,
}


# Union type for annotations.
Component = Union[
    SersicComponent, DevaucComponent, ExpdiskComponent, EdgediskComponent,
    GaussianComponent, MoffatComponent, NukerComponent, KingComponent,
    FerrerComponent, PsfComponent, SkyComponent, TruncationComponent,
]


# =============================================================================
# Section 5: GalfitHeader and GalfitFile
# =============================================================================

@dataclass
class GalfitHeader:
    input_image: Optional[str] = None                       # A
    output_block: Optional[str] = None                      # B
    sigma_image: Optional[str] = None                       # C
    psf_image: Optional[str] = None                         # D
    psf_diffusion_kernel: Optional[str] = None              # D (second token)
    psf_fine_sampling: int = 1                              # E
    bad_pixel_mask: Optional[str] = None                    # F
    constraint_file: Optional[str] = None                   # G
    fit_region: Optional[tuple[int, int, int, int]] = None  # H
    conv_box: Optional[tuple[int, int]] = None              # I
    zeropoint: Optional[float] = None                       # J
    plate_scale: Optional[tuple[float, float]] = None       # K
    display_type: str = "regular"                           # O
    options: int = 0                                        # P


@dataclass
class GalfitFile:
    header: GalfitHeader = field(default_factory=GalfitHeader)
    components: list[_ComponentBase] = field(default_factory=list)
    # Freeform leading comment lines preserved verbatim so output files that
    # start with "#  Input menu file:  galfit.01" etc. round-trip.
    leading_comments: list[str] = field(default_factory=list)


# =============================================================================
# Section 6: Low-level tokenizer
# =============================================================================

_KEY_LINE_RE = re.compile(r"^\s*(\w+)\)\s*(.*?)\s*$")


@dataclass
class _KeyLine:
    key: str                # e.g. "A", "H", "F1", "R0", "Ti", "C0"
    raw: str                # original line (for error messages)
    payload_tokens: list[str]
    comment: Optional[str]


def _split_comment(line: str) -> tuple[str, Optional[str]]:
    """Split off a trailing ``# comment`` if any. A `#` inside parenthesized
    uncertainty tuples is not possible in GALFIT output, so we can safely
    split on the first `#` after at least one whitespace."""
    if "#" not in line:
        return line.rstrip("\n"), None
    # Find the first # that appears after whitespace or at line start.
    body, _, comment = line.partition("#")
    return body.rstrip(), comment.rstrip("\n").rstrip()


def _gather_value_tokens(payload: str) -> list[str]:
    """Tokenize a payload string, keeping parenthesized (err [err ...]) tuples
    as a single token."""
    raw = payload.split()
    tokens: list[str] = []
    buf: list[str] = []
    in_paren = False
    for t in raw:
        if in_paren:
            buf.append(t)
            if t.endswith(")"):
                tokens.append(" ".join(buf))
                buf = []
                in_paren = False
        else:
            if t.startswith("(") and not t.endswith(")"):
                buf = [t]
                in_paren = True
            elif t.startswith("(") and t.endswith(")"):
                tokens.append(t)
            else:
                tokens.append(t)
    if buf:
        # Unbalanced paren: emit the buffered content as a best-effort token.
        tokens.append(" ".join(buf))
    return tokens


def _tokenize(text: str) -> tuple[list[str], list[_KeyLine]]:
    """Return (leading_comments, key_lines). Comments preceding the first
    keyed line are preserved; inline key-line comments are attached per
    _KeyLine.comment."""
    leading: list[str] = []
    keyed: list[_KeyLine] = []
    seen_key = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            if not seen_key:
                leading.append(line)
            continue
        body, comment = _split_comment(line)
        m = _KEY_LINE_RE.match(body)
        if not m:
            logger.debug("ignoring unparseable line: %r", raw_line)
            continue
        key = m.group(1)
        payload = m.group(2)
        tokens = _gather_value_tokens(payload)
        keyed.append(_KeyLine(key=key, raw=raw_line, payload_tokens=tokens, comment=comment))
        seen_key = True
    return leading, keyed


# =============================================================================
# Section 7: Value parsing (envelopes, uncertainties, fit flags)
# =============================================================================

def _parse_paren_errors(token: str) -> list[float]:
    """Strip surrounding parens and return the space-separated floats."""
    inner = token.strip()
    if not (inner.startswith("(") and inner.endswith(")")):
        raise ValueError(f"not a parenthesized tuple: {token!r}")
    inner = inner[1:-1]
    return [float(x) for x in inner.split() if x]


def _consume_fitted_values(
    tokens: Sequence[str], n_values: int,
) -> tuple[list[FittedValue], list[str]]:
    """Consume n_values FittedValue objects from the head of ``tokens``.

    Layout accepted:
        v1 [v2 ...] [(err1 err2 ...)] flag [flag ...] [remaining tokens]

    The uncertainty tuple is optional (input files won't have it). The fit
    flags column tolerates either n_values flags (one per value) or a
    single flag (broadcast to all values) - the local EXAMPLE.INPUT file
    ships a truncation-attached ``F1) 0.1 30 1`` row that relies on the
    broadcast form.
    """
    if len(tokens) < n_values + 1:
        raise ValueError(
            f"need at least {n_values + 1} tokens for {n_values}-value row, "
            f"got {list(tokens)}"
        )
    values_raw = list(tokens[:n_values])
    cursor = n_values
    uncertainties: list[Optional[float]] = [None] * n_values
    if cursor < len(tokens) and tokens[cursor].startswith("("):
        errs = _parse_paren_errors(tokens[cursor])
        if len(errs) != n_values:
            raise ValueError(
                f"uncertainty tuple has {len(errs)} entries but row has {n_values} values: "
                f"{tokens[cursor]!r}"
            )
        uncertainties = [float(e) for e in errs]
        cursor += 1
    remaining = len(tokens) - cursor
    if remaining <= 0:
        raise ValueError(
            f"missing fit flags after values/uncertainties: {list(tokens)}"
        )
    if remaining >= n_values:
        fit_flags = [t == "1" for t in tokens[cursor:cursor + n_values]]
        cursor += n_values
    else:
        # Broadcast a single flag to all values.
        single_flag = tokens[cursor] == "1"
        fit_flags = [single_flag] * n_values
        cursor += 1
    fvs: list[FittedValue] = []
    for raw, unc, fit in zip(values_raw, uncertainties, fit_flags):
        inner, fixed, constr, susp = _strip_envelope(raw)
        fvs.append(FittedValue(
            value=_parse_number(inner),
            fit=fit,
            uncertainty=unc,
            fixed_in_output=fixed,
            constrained_in_output=constr,
            suspicious=susp,
        ))
    return fvs, list(tokens[cursor:])


# =============================================================================
# Section 8: Header parsing
# =============================================================================

def _parse_header_line(header: GalfitHeader, key: str, tokens: list[str]) -> None:
    if key not in HEADER_SCHEMA:
        return
    attr, kind = HEADER_SCHEMA[key]
    if kind == "str1":
        setattr(header, attr, tokens[0] if tokens else None)
    elif kind == "int1":
        setattr(header, attr, int(tokens[0]))
    elif kind == "flt1":
        setattr(header, attr, float(tokens[0]))
    elif kind == "int4":
        if len(tokens) < 4:
            raise ValueError(f"header {key}) expects 4 ints, got {tokens}")
        setattr(header, attr, tuple(int(t) for t in tokens[:4]))
    elif kind == "int2":
        if len(tokens) < 2:
            raise ValueError(f"header {key}) expects 2 ints, got {tokens}")
        setattr(header, attr, tuple(int(t) for t in tokens[:2]))
    elif kind == "flt2":
        if len(tokens) < 2:
            raise ValueError(f"header {key}) expects 2 floats, got {tokens}")
        setattr(header, attr, tuple(float(t) for t in tokens[:2]))
    elif kind == "psf":
        header.psf_image = tokens[0] if tokens else None
        header.psf_diffusion_kernel = tokens[1] if len(tokens) >= 2 else None


# =============================================================================
# Section 9: Component parsing
# =============================================================================

def _resolve_profile_token(token: str) -> tuple[str, Optional[int]]:
    """Accept sersic | sersic1 | sersic2 | sersic3 (and variants for
    other profiles). Return (canonical_name, suffix_or_None)."""
    token = token.lower()
    if token in VALID_PROFILES:
        return token, None
    if token[-1].isdigit():
        base = token[:-1]
        suffix = int(token[-1])
        if base in VALID_PROFILES and suffix in VALID_FLUX_NORM_SUFFIXES:
            return base, suffix
    raise ValueError(f"unknown profile type: {token!r}")


def _new_component(profile: str, suffix: Optional[int]) -> _ComponentBase:
    cls = PROFILE_REGISTRY[profile]
    comp = cls()
    comp.flux_norm_suffix = suffix
    return comp


def _assign_numeric_param(
    component: _ComponentBase, param_num: int, fv: FittedValue,
) -> None:
    schema = component.PARAM_SCHEMA
    if param_num in schema:
        attr, _ = schema[param_num]
        setattr(component, attr, fv)
        return
    # Legacy pre-3.0: parameter 8 was axis ratio. Post-3.0 GALFIT
    # emits 8) 0.0 0 as a reserved placeholder. Distinguish by checking
    # whether the value is non-zero or the fit flag is set.
    if param_num == 8 and 9 in schema and (fv.value != 0.0 or fv.fit):
        logger.warning(
            "%s component uses legacy pre-3.0 parameter 8) for axis ratio; "
            "mapping to 9) q.", component.profile,
        )
        attr, _ = schema[9]
        setattr(component, attr, fv)
        return
    # Unknown / reserved slot - absorb so restart files round-trip.
    logger.debug(
        "%s component: param %d) not in schema; storing as extra_param",
        component.profile, param_num,
    )
    component.extra_params[param_num] = fv


def _apply_hidden_key(
    component: _ComponentBase, key: str, tokens: list[str],
) -> None:
    """Apply a hidden-block key (Z, C0, B<N>, F<N>, R0..R10, Ti, To) to the
    current component's `hidden` field."""
    hb = component.hidden
    # Z) integer, no fit flag.
    if key == "Z":
        hb.z_skip = int(tokens[0]) if tokens else 0
        return
    # C0) single value + fit flag.
    if key == "C0":
        fvs, _ = _consume_fitted_values(tokens, 1)
        hb.c0 = fvs[0]
        return
    # Ti) and To) single integer, no fit flag.
    if key == "Ti":
        component.hidden.trunc_inner_ref = int(tokens[0])
        return
    if key == "To":
        component.hidden.trunc_outer_ref = int(tokens[0])
        return
    # B<N>) single value + fit flag.
    if key.startswith("B") and key[1:].isdigit():
        idx = int(key[1:])
        fvs, _ = _consume_fitted_values(tokens, 1)
        hb.bending[idx] = fvs[0]
        return
    # F<N>) two values + two fit flags.
    if key.startswith("F") and key[1:].isdigit():
        idx = int(key[1:])
        fvs, _ = _consume_fitted_values(tokens, 2)
        hb.fourier[idx] = (fvs[0], fvs[1])
        return
    # R0) is an enum string, the rest are numeric + fit flag.
    if key == "R0":
        mode = tokens[0].lower() if tokens else "none"
        _warn_rotation_mode(mode)
        if hb.rotation is None:
            hb.rotation = RotationBlock(mode=mode)
        else:
            hb.rotation.mode = mode
        return
    if key.startswith("R") and key[1:].isdigit():
        if hb.rotation is None:
            hb.rotation = RotationBlock()
        idx = int(key[1:])
        fvs, _ = _consume_fitted_values(tokens, 1)
        _assign_rotation_slot(hb.rotation, idx, fvs[0])
        return


def _assign_rotation_slot(block: RotationBlock, idx: int, fv: FittedValue) -> None:
    mapping = {
        1: "r_in", 2: "r_out", 3: "theta_out", 4: "alpha_or_rws",
        9: "incl", 10: "sky_pa",
    }
    if idx in mapping:
        setattr(block, mapping[idx], fv)
    else:
        block.extra[idx] = fv


def _warn_rotation_mode(mode: str) -> None:
    if mode in ROTATION_TYPES_DOCUMENTED:
        return
    if mode in ROTATION_TYPES_ASPIRATIONAL:
        logger.warning(
            "rotation mode %r is mentioned in EXAMPLE.INPUT but not "
            "documented in README (%s); GALFIT may reject it.",
            mode, ", ".join(ROTATION_TYPES_DOCUMENTED),
        )
    else:
        logger.warning("unknown rotation mode %r", mode)


def _parse_component_block(
    lines: Sequence[_KeyLine],
) -> _ComponentBase:
    """Parse one component block. The first line must be 0) or T0)."""
    first = lines[0]
    if first.key == "0":
        profile, suffix = _resolve_profile_token(first.payload_tokens[0])
        component: _ComponentBase = _new_component(profile, suffix)
        body = lines[1:]
    elif first.key == "T0":
        trunc_type = first.payload_tokens[0].lower() if first.payload_tokens else "radial"
        if trunc_type not in TRUNCATION_TYPES:
            logger.warning("unknown truncation type %r", trunc_type)
        component = TruncationComponent(trunc_type=trunc_type)
        body = lines[1:]
    else:
        raise ValueError(
            f"component block must start with 0) or T0); got {first.key}) on "
            f"{first.raw!r}"
        )

    for kl in body:
        key = kl.key
        tokens = kl.payload_tokens
        if key == "1":
            n_val = 1 if isinstance(component, SkyComponent) else 2
            fvs, _ = _consume_fitted_values(tokens, n_val)
            if isinstance(component, SkyComponent):
                component.dc = fvs[0]
            else:
                component.x, component.y = fvs[0], fvs[1]
            continue
        if key == "T1" and isinstance(component, TruncationComponent):
            fvs, _ = _consume_fitted_values(tokens, 2)
            component.x, component.y = fvs[0], fvs[1]
            continue
        if key.isdigit():
            param_num = int(key)
            fvs, _ = _consume_fitted_values(tokens, 1)
            _assign_numeric_param(component, param_num, fvs[0])
            continue
        if (key.startswith("T") and len(key) > 1 and key[1:].isdigit()
                and isinstance(component, TruncationComponent)):
            param_num = int(key[1:])
            fvs, _ = _consume_fitted_values(tokens, 1)
            _assign_numeric_param(component, param_num, fvs[0])
            continue
        # Everything else is a hidden block key.
        _apply_hidden_key(component, key, tokens)
    return component


# =============================================================================
# Section 10: Block splitting and full-file parser
# =============================================================================

_HEADER_KEYS_SET = set(HEADER_SCHEMA.keys()) | {"S"}  # S) is interactive mode
_BLOCK_STARTER_KEYS = {"0", "T0"}


def _partition_blocks(
    keyed: list[_KeyLine],
) -> tuple[list[_KeyLine], list[list[_KeyLine]]]:
    header_lines: list[_KeyLine] = []
    blocks: list[list[_KeyLine]] = []
    current: list[_KeyLine] = []
    for kl in keyed:
        if kl.key in _HEADER_KEYS_SET and not current:
            header_lines.append(kl)
            continue
        if kl.key in _BLOCK_STARTER_KEYS:
            if current:
                blocks.append(current)
            current = [kl]
        else:
            if not current:
                # A stray hidden-block key before any component - unusual;
                # drop with a warning. Real GALFIT would error out here.
                logger.warning("ignoring %s) before any component block", kl.key)
                continue
            current.append(kl)
    if current:
        blocks.append(current)
    return header_lines, blocks


def read_galfit_text(text: str) -> GalfitFile:
    """Parse a GALFIT config/output text into a GalfitFile."""
    leading, keyed = _tokenize(text)
    header_lines, blocks = _partition_blocks(keyed)
    header = GalfitHeader()
    for kl in header_lines:
        if kl.key == "S":
            continue
        _parse_header_line(header, kl.key, kl.payload_tokens)
    components = [_parse_component_block(b) for b in blocks]
    return GalfitFile(
        header=header,
        components=components,
        leading_comments=leading,
    )


def read_galfit(path: Union[str, Path]) -> GalfitFile:
    """Parse a GALFIT config/output file."""
    return read_galfit_text(Path(path).read_text(encoding="utf-8"))


# =============================================================================
# Section 11: Writers
# =============================================================================

_HEADER_COMMENTS: dict[str, str] = {
    "A": "Input data image (FITS file)",
    "B": "Output data image block",
    "C": "Sigma image name (made from data if blank or \"none\")",
    "D": "Input PSF image and (optional) diffusion kernel",
    "E": "PSF fine sampling factor relative to data",
    "F": "Bad pixel mask (FITS image or ASCII coord list)",
    "G": "File with parameter constraints (ASCII file)",
    "H": "Image region to fit (xmin xmax ymin ymax)",
    "I": "Size of the convolution box (x y)",
    "J": "Magnitude photometric zeropoint",
    "K": "Plate scale (dx dy) [arcsec per pixel]",
    "O": "Display type (regular, curses, both)",
    "P": "Options: 0=normal run; 1,2=make model/imgblock & quit",
}


def _format_fit_flag(fit: bool) -> str:
    return "1" if fit else "0"


def _format_value_line(
    key: str,
    fitted: Sequence[FittedValue],
    comment: str,
    *,
    preserve_decorations: bool,
) -> str:
    value_toks = [fv.to_token(preserve_decorations=preserve_decorations)
                  for fv in fitted]
    line = f"{key:>3}) " + " ".join(value_toks)
    if preserve_decorations and any(fv.uncertainty is not None for fv in fitted):
        errs = " ".join(
            _format_number(fv.uncertainty) if fv.uncertainty is not None else "0"
            for fv in fitted
        )
        line += f" ({errs})"
    line += " " + " ".join(_format_fit_flag(fv.fit) for fv in fitted)
    if comment:
        line += f"    # {comment}"
    return line


def _write_header_lines(header: GalfitHeader) -> list[str]:
    out: list[str] = []
    out.append("=" * 79)
    for key, (attr, kind) in HEADER_SCHEMA.items():
        value = getattr(header, attr, None)
        if value is None and kind != "psf":
            continue
        if kind == "str1":
            tok = value
        elif kind == "int1":
            tok = str(value)
        elif kind == "flt1":
            tok = _format_number(float(value))
        elif kind == "int4":
            tok = " ".join(str(int(v)) for v in value)
        elif kind == "int2":
            tok = " ".join(str(int(v)) for v in value)
        elif kind == "flt2":
            tok = " ".join(_format_number(float(v)) for v in value)
        elif kind == "psf":
            tokens = []
            if header.psf_image is not None:
                tokens.append(header.psf_image)
            if header.psf_diffusion_kernel is not None:
                tokens.append(header.psf_diffusion_kernel)
            if not tokens:
                continue
            tok = " ".join(tokens)
        else:
            raise AssertionError(f"unknown header kind {kind}")
        out.append(f"{key}) {tok:<22} # {_HEADER_COMMENTS[key]}")
    out.append("")
    return out


def _write_component_block(
    component: _ComponentBase, index: int, *, preserve_decorations: bool,
) -> list[str]:
    out: list[str] = []
    out.append(f"# Component number: {index}")
    if isinstance(component, TruncationComponent):
        out.append(_format_scalar_line("T0", component.trunc_type,
                                       f"truncation type"))
        out.append(_format_value_line(
            "T1", [component.x, component.y], "position x, y [pixel]",
            preserve_decorations=preserve_decorations,
        ))
        for num in sorted(component.PARAM_SCHEMA):
            attr, label = component.PARAM_SCHEMA[num]
            fv = getattr(component, attr)
            if fv is None:
                continue
            out.append(_format_value_line(
                f"T{num}", [fv], label,
                preserve_decorations=preserve_decorations,
            ))
        for num in sorted(component.extra_params):
            out.append(_format_value_line(
                f"T{num}", [component.extra_params[num]], f"T{num} (unlabelled)",
                preserve_decorations=preserve_decorations,
            ))
    else:
        out.append(f" 0) {component.profile_token:<16} # Object type")
        if isinstance(component, SkyComponent):
            out.append(_format_value_line(
                "1", [component.dc], "sky background DC [ADU counts]",
                preserve_decorations=preserve_decorations,
            ))
            out.append(_format_value_line(
                "2", [component.dsky_dx], "dsky/dx (sky gradient in x)",
                preserve_decorations=preserve_decorations,
            ))
            out.append(_format_value_line(
                "3", [component.dsky_dy], "dsky/dy (sky gradient in y)",
                preserve_decorations=preserve_decorations,
            ))
        else:
            out.append(_format_value_line(
                "1", [component.x, component.y], "position x, y [pixel]",
                preserve_decorations=preserve_decorations,
            ))
            for num in sorted(component.PARAM_SCHEMA):
                attr, label = component.PARAM_SCHEMA[num]
                fv = getattr(component, attr)
                out.append(_format_value_line(
                    str(num), [fv], label,
                    preserve_decorations=preserve_decorations,
                ))
    out.extend(_write_hidden_blocks(
        component.hidden, preserve_decorations=preserve_decorations,
    ))
    out.append("")
    return out


def _format_scalar_line(key: str, value: str, comment: str) -> str:
    line = f"{key:>3}) {value:<16}"
    if comment:
        line += f" # {comment}"
    return line


def _write_hidden_blocks(
    hb: HiddenBlocks, *, preserve_decorations: bool,
) -> list[str]:
    out: list[str] = []
    if hb.c0 is not None:
        out.append(_format_value_line(
            "C0", [hb.c0], "diskyness(-)/boxyness(+)",
            preserve_decorations=preserve_decorations,
        ))
    for n in sorted(hb.bending):
        out.append(_format_value_line(
            f"B{n}", [hb.bending[n]], f"bending mode {n}",
            preserve_decorations=preserve_decorations,
        ))
    for n in sorted(hb.fourier):
        amp, phase = hb.fourier[n]
        out.append(_format_value_line(
            f"F{n}", [amp, phase],
            f"Az. Fourier mode {n}: amplitude, phase angle [deg]",
            preserve_decorations=preserve_decorations,
        ))
    if hb.rotation is not None:
        out.append(_format_scalar_line("R0", hb.rotation.mode,
                                       "PA rotation function"))
        r_slots = {
            1: ("r_in", "bar radius [pixels]"),
            2: ("r_out", "asymptotic radius [pixels]"),
            3: ("theta_out", "cumulative rotation to asymp. radius [deg]"),
            4: ("alpha_or_rws", "powerlaw alpha or log winding scale"),
            9: ("incl", "inclination to line of sight [deg]"),
            10: ("sky_pa", "sky position angle [deg]"),
        }
        for num, (attr, label) in sorted(r_slots.items()):
            fv = getattr(hb.rotation, attr)
            if fv is None:
                continue
            out.append(_format_value_line(
                f"R{num}", [fv], label,
                preserve_decorations=preserve_decorations,
            ))
        for num in sorted(hb.rotation.extra):
            out.append(_format_value_line(
                f"R{num}", [hb.rotation.extra[num]], f"R{num}",
                preserve_decorations=preserve_decorations,
            ))
    if hb.trunc_inner_ref is not None:
        out.append(f"Ti) {hb.trunc_inner_ref}    # inner truncation by component")
    if hb.trunc_outer_ref is not None:
        out.append(f"To) {hb.trunc_outer_ref}    # outer truncation by component")
    # Always emit Z - GALFIT expects it at the end of every component.
    out.append(f" Z) {hb.z_skip}                  # leave in [1] or subtract [0]")
    return out


def write_galfit_text(
    galfit_file: GalfitFile, *, preserve_decorations: bool = False,
) -> str:
    lines: list[str] = []
    if galfit_file.leading_comments:
        lines.extend(galfit_file.leading_comments)
        lines.append("")
    lines.extend(_write_header_lines(galfit_file.header))
    for idx, comp in enumerate(galfit_file.components, start=1):
        lines.extend(_write_component_block(
            comp, idx, preserve_decorations=preserve_decorations,
        ))
    return "\n".join(lines) + "\n"


def write_galfit(
    galfit_file: GalfitFile, path: Union[str, Path],
    *, preserve_decorations: bool = False,
) -> None:
    Path(path).write_text(
        write_galfit_text(galfit_file, preserve_decorations=preserve_decorations),
        encoding="utf-8",
    )


# =============================================================================
# Section 12: Dict / YAML / JSON serialization
# =============================================================================

def _dataclass_to_dict(obj: Any) -> Any:
    if isinstance(obj, FittedValue):
        return obj.to_dict()
    if isinstance(obj, RotationBlock):
        d: dict[str, Any] = {"mode": obj.mode}
        for name in ("r_in", "r_out", "theta_out", "alpha_or_rws", "incl", "sky_pa"):
            v = getattr(obj, name)
            if v is not None:
                d[name] = _dataclass_to_dict(v)
        if obj.extra:
            d["extra"] = {str(k): _dataclass_to_dict(v) for k, v in obj.extra.items()}
        return d
    if isinstance(obj, HiddenBlocks):
        d = {"z_skip": obj.z_skip}
        if obj.c0 is not None:
            d["c0"] = _dataclass_to_dict(obj.c0)
        if obj.bending:
            d["bending"] = {str(k): _dataclass_to_dict(v) for k, v in obj.bending.items()}
        if obj.fourier:
            d["fourier"] = {
                str(k): [_dataclass_to_dict(a), _dataclass_to_dict(b)]
                for k, (a, b) in obj.fourier.items()
            }
        if obj.rotation is not None:
            d["rotation"] = _dataclass_to_dict(obj.rotation)
        if obj.trunc_inner_ref is not None:
            d["trunc_inner_ref"] = obj.trunc_inner_ref
        if obj.trunc_outer_ref is not None:
            d["trunc_outer_ref"] = obj.trunc_outer_ref
        return d
    if isinstance(obj, TruncationComponent):
        d = {"profile": "__truncation__", "trunc_type": obj.trunc_type}
        d["x"] = _dataclass_to_dict(obj.x)
        d["y"] = _dataclass_to_dict(obj.y)
        for num in sorted(obj.PARAM_SCHEMA):
            attr, _ = obj.PARAM_SCHEMA[num]
            v = getattr(obj, attr)
            if v is not None:
                d[attr] = _dataclass_to_dict(v)
        if obj.extra_params:
            d["extra_params"] = {
                str(k): _dataclass_to_dict(v) for k, v in obj.extra_params.items()
            }
        d["hidden"] = _dataclass_to_dict(obj.hidden)
        return d
    if isinstance(obj, _ComponentBase):
        d = {"profile": obj.profile}
        if obj.flux_norm_suffix is not None:
            d["flux_norm_suffix"] = obj.flux_norm_suffix
        if not isinstance(obj, SkyComponent):
            d["x"] = _dataclass_to_dict(obj.x)
            d["y"] = _dataclass_to_dict(obj.y)
        for num in sorted(obj.PARAM_SCHEMA):
            attr, _ = obj.PARAM_SCHEMA[num]
            d[attr] = _dataclass_to_dict(getattr(obj, attr))
        if obj.extra_params:
            d["extra_params"] = {
                str(k): _dataclass_to_dict(v) for k, v in obj.extra_params.items()
            }
        d["hidden"] = _dataclass_to_dict(obj.hidden)
        return d
    if isinstance(obj, GalfitHeader):
        d = {}
        for f in fields(obj):
            v = getattr(obj, f.name)
            if v is None:
                continue
            if isinstance(v, tuple):
                d[f.name] = list(v)
            else:
                d[f.name] = v
        return d
    if isinstance(obj, GalfitFile):
        return {
            "header": _dataclass_to_dict(obj.header),
            "components": [_dataclass_to_dict(c) for c in obj.components],
            "leading_comments": list(obj.leading_comments),
        }
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def to_dict(galfit_file: GalfitFile) -> dict:
    return _dataclass_to_dict(galfit_file)


def from_dict(data: dict) -> GalfitFile:
    header_data = data.get("header", {})
    header = GalfitHeader()
    for key, value in header_data.items():
        if key in {"fit_region", "conv_box", "plate_scale"} and isinstance(value, list):
            value = tuple(value)
        setattr(header, key, value)
    components: list[_ComponentBase] = []
    for comp_data in data.get("components", []):
        components.append(_component_from_dict(comp_data))
    return GalfitFile(
        header=header,
        components=components,
        leading_comments=list(data.get("leading_comments", [])),
    )


def _component_from_dict(data: dict) -> _ComponentBase:
    profile = data["profile"]
    if profile == "__truncation__":
        comp = TruncationComponent(trunc_type=data.get("trunc_type", "radial"))
        comp.x = FittedValue.from_dict(data.get("x", 0.0))
        comp.y = FittedValue.from_dict(data.get("y", 0.0))
        for num in sorted(comp.PARAM_SCHEMA):
            attr, _ = comp.PARAM_SCHEMA[num]
            if attr in data:
                setattr(comp, attr, FittedValue.from_dict(data[attr]))
        for k, v in (data.get("extra_params") or {}).items():
            comp.extra_params[int(k)] = FittedValue.from_dict(v)
        comp.hidden = _hidden_from_dict(data.get("hidden", {}))
        return comp
    cls = PROFILE_REGISTRY[profile]
    comp = cls()
    comp.flux_norm_suffix = data.get("flux_norm_suffix")
    if "x" in data:
        comp.x = FittedValue.from_dict(data["x"])
    if "y" in data:
        comp.y = FittedValue.from_dict(data["y"])
    for num in sorted(comp.PARAM_SCHEMA):
        attr, _ = comp.PARAM_SCHEMA[num]
        if attr in data:
            setattr(comp, attr, FittedValue.from_dict(data[attr]))
    for k, v in (data.get("extra_params") or {}).items():
        comp.extra_params[int(k)] = FittedValue.from_dict(v)
    comp.hidden = _hidden_from_dict(data.get("hidden", {}))
    return comp


def _hidden_from_dict(data: dict) -> HiddenBlocks:
    hb = HiddenBlocks()
    hb.z_skip = int(data.get("z_skip", 0))
    if "c0" in data:
        hb.c0 = FittedValue.from_dict(data["c0"])
    for k, v in (data.get("bending") or {}).items():
        hb.bending[int(k)] = FittedValue.from_dict(v)
    for k, v in (data.get("fourier") or {}).items():
        amp, phase = v
        hb.fourier[int(k)] = (
            FittedValue.from_dict(amp),
            FittedValue.from_dict(phase),
        )
    if "rotation" in data:
        r = data["rotation"]
        block = RotationBlock(mode=r.get("mode", "none"))
        for name in ("r_in", "r_out", "theta_out", "alpha_or_rws", "incl", "sky_pa"):
            if name in r:
                setattr(block, name, FittedValue.from_dict(r[name]))
        for k, v in (r.get("extra") or {}).items():
            block.extra[int(k)] = FittedValue.from_dict(v)
        hb.rotation = block
    if "trunc_inner_ref" in data:
        hb.trunc_inner_ref = int(data["trunc_inner_ref"])
    if "trunc_outer_ref" in data:
        hb.trunc_outer_ref = int(data["trunc_outer_ref"])
    return hb


def to_yaml(galfit_file: GalfitFile) -> str:
    import yaml as _yaml  # local import to keep the top-level try/except clean
    return _yaml.safe_dump(to_dict(galfit_file), sort_keys=False)


def from_yaml(text: str) -> GalfitFile:
    import yaml as _yaml
    return from_dict(_yaml.safe_load(text))


def to_json(galfit_file: GalfitFile, *, indent: int = 2) -> str:
    return json.dumps(to_dict(galfit_file), indent=indent)


def from_json(text: str) -> GalfitFile:
    return from_dict(json.loads(text))
