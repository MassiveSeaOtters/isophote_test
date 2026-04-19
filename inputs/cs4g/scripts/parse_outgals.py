#!/usr/bin/env python
"""
parse_outgals.py - Parse downloaded `.outgal` files into structured records.

Uses the bundled `/galfit` skill's parser (`~/.claude/skills/galfit/`) to
read each cached `.outgal`, extracts per-component structural parameters,
maps the column-3 semantics correctly per profile (integrated mag for
sersic/expdisk/devauc/psf/etc., mu0 for edgedisk, mu_rb for nuker), pulls
out the S4G `# STRUCTURE:` labels for each component, and tags any
edge-disk-bearing galaxy for downstream skip.

Output: `inputs/cs4g/cs4g_models.json`, keyed by galaxy name. Skip-listed
galaxies are written to `inputs/cs4g/cs4g_skipped.json`.

Usage:
    python inputs/cs4g/scripts/parse_outgals.py --limit 10
    python inputs/cs4g/scripts/parse_outgals.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = REPO_ROOT / "inputs" / "cs4g" / "cs4g_p4_index.csv"
DEFAULT_MIRROR = REPO_ROOT / "inputs" / "cs4g" / "p4"
DEFAULT_MODELS = REPO_ROOT / "inputs" / "cs4g" / "cs4g_models.json"
DEFAULT_SKIPPED = REPO_ROOT / "inputs" / "cs4g" / "cs4g_skipped.json"

# Profile types where col `3)` is the INTEGRATED magnitude (drop-in friendly)
MAG_PROFILES = {"sersic", "devauc", "expdisk", "gaussian", "moffat", "psf"}
# Profile types where col `3)` is a SURFACE BRIGHTNESS (mag/arcsec^2).
# Each needs profile-specific integration to recover total mag, deferred to
# Phase 5. For now we drop galaxies that contain any of these.
# Attribute name on the parsed component, per the /galfit skill schema.
SB_PROFILE_ATTR = {
    "edgedisk": "mu0",        # central SB
    "nuker":    "mu_rb",      # SB at break radius
    "king":     "mu0",        # central SB
    "ferrer":   "mu_central", # central SB
}
SB_PROFILES = set(SB_PROFILE_ATTR)

# Reasons we drop a whole galaxy from the mock-input pool. Currently
# only edgedisk: libprofit cannot render it, period. nuker/king/ferrer
# are kept and their SB values are converted to integrated mag in Phase 5.
SKIP_PROFILES = {
    "edgedisk":  "has_edgedisk",
}

# S4G pipeline annotation: "# STRUCTURE: BULGE" / "# STRUCTURE: ZDISK" etc.
_STRUCTURE_RE = re.compile(r"^\s*#\s*STRUCTURE:\s*(\S+)", re.IGNORECASE)
_COMP_NUMBER_RE = re.compile(r"^\s*#\s*Component number:\s*(\d+)", re.IGNORECASE)


def import_galfit_skill():
    """Import the parser from the user's installed /galfit skill."""
    skill_path = Path.home() / ".claude" / "skills" / "galfit" / "scripts"
    if not skill_path.exists():
        raise RuntimeError(
            f"Galfit skill not found at {skill_path}. "
            "Install via the /galfit SKILL package."
        )
    if str(skill_path) not in sys.path:
        sys.path.insert(0, str(skill_path))
    import galfit_io  # noqa: E402
    return galfit_io


def extract_structure_labels(text: str) -> dict[int, str]:
    """Map component number (1-based) -> STRUCTURE label (or '' if absent)."""
    labels: dict[int, str] = {}
    current_comp: int | None = None
    for line in text.splitlines():
        m = _COMP_NUMBER_RE.match(line)
        if m:
            current_comp = int(m.group(1))
            labels.setdefault(current_comp, "")
            continue
        m = _STRUCTURE_RE.match(line)
        if m and current_comp is not None:
            labels[current_comp] = m.group(1)
    return labels


def component_to_record(comp, profile: str) -> dict:
    """Extract a flat record from a galfit component, preserving col-3 semantics."""
    rec: dict = {"profile": profile}
    # GALFIT's flux-norm suffix (e.g. ferrer2, sersic1). Salo+2015 uses
    # ferrer2 for bars; the integrator in Phase 5 needs to know the suffix
    # because col-3 means different surface-brightness flavors per variant.
    if comp.flux_norm_suffix is not None:
        rec["flux_norm_suffix"] = comp.flux_norm_suffix
    if profile != "sky":
        rec["x_pix"] = float(comp.x.value)
        rec["y_pix"] = float(comp.y.value)

    # Col 3) — branches by profile type
    if profile in MAG_PROFILES:
        rec["mag_app"] = float(comp.mag.value)
        rec["col3_kind"] = "integrated_mag"
    elif profile in SB_PROFILES:
        attr = SB_PROFILE_ATTR[profile]
        rec[attr] = float(getattr(comp, attr).value)
        rec["col3_kind"] = attr
    elif profile == "sky":
        rec["sky_dc"] = float(comp.dc.value)
        rec["sky_dx"] = float(comp.dsky_dx.value)
        rec["sky_dy"] = float(comp.dsky_dy.value)
        rec["col3_kind"] = "sky"
    else:
        rec["col3_kind"] = "unknown"

    # Geometric scales — branch per profile
    if profile in {"sersic", "devauc"}:
        rec["re_pix"] = float(comp.re.value)
        if profile == "sersic":
            rec["sersic_n"] = float(comp.n.value)
    elif profile == "expdisk":
        rec["rs_pix"] = float(comp.rs.value)
    elif profile in {"gaussian", "moffat"}:
        rec["fwhm_pix"] = float(comp.fwhm.value)
        if profile == "moffat":
            rec["moffat_n"] = float(comp.n.value)
    elif profile == "edgedisk":
        rec["h_s_pix"] = float(comp.h_s.value)
        rec["r_s_pix"] = float(comp.r_s.value)
    elif profile == "nuker":
        rec["rb_pix"] = float(comp.rb.value)
        rec["alpha"] = float(comp.alpha.value)
        rec["beta"] = float(comp.beta.value)
        rec["gamma"] = float(comp.gamma.value)
    elif profile == "king":
        rec["rc_pix"] = float(comp.rc.value)
        rec["rt_pix"] = float(comp.rt.value)
        rec["alpha"] = float(comp.alpha.value)
    elif profile == "ferrer":
        rec["r_out_pix"] = float(comp.r_out.value)
        rec["alpha"] = float(comp.alpha.value)
        rec["beta"] = float(comp.beta.value)

    # Geometry: q (axis ratio) and PA — common where applicable
    if hasattr(comp, "q"):
        rec["q"] = float(comp.q.value)
    if hasattr(comp, "pa") and profile != "sky":
        rec["pa_deg"] = float(comp.pa.value)
    return rec


def parse_one(path: Path, galfit_io) -> tuple[dict | None, str | None]:
    """Parse one outgal. Returns (model_record, skip_reason).

    skip_reason is non-None when the galaxy should be excluded
    (currently: any edgedisk component present).
    """
    text = path.read_text()
    structure_labels = extract_structure_labels(text)
    try:
        gf = galfit_io.read_galfit(path)
    except Exception as e:
        return None, f"parse_error: {e}"

    components = []
    skip_reason: str | None = None
    for i, comp in enumerate(gf.components, start=1):
        profile = comp.profile
        if profile in SKIP_PROFILES and skip_reason is None:
            skip_reason = SKIP_PROFILES[profile]
        rec = component_to_record(comp, profile)
        rec["component_number"] = i
        rec["structure_label"] = structure_labels.get(i, "")
        components.append(rec)

    if skip_reason is not None:
        return None, skip_reason

    record = {
        "outgal_path": str(path.relative_to(REPO_ROOT)),
        "input_image": gf.header.input_image,
        "zeropoint": gf.header.zeropoint,
        "plate_scale_arcsec_per_pix": (gf.header.plate_scale[0]
                                       if gf.header.plate_scale else None),
        "fit_region": list(gf.header.fit_region) if gf.header.fit_region else None,
        "n_components": len(components),
        "n_light_components": sum(1 for c in components if c["profile"] != "sky"),
        "components": components,
    }
    return record, None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                   help=f"P4 index CSV (default: {DEFAULT_INDEX})")
    p.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR,
                   help=f"Local outgal mirror (default: {DEFAULT_MIRROR})")
    p.add_argument("--out-models", type=Path, default=DEFAULT_MODELS,
                   help=f"Output models JSON (default: {DEFAULT_MODELS})")
    p.add_argument("--out-skipped", type=Path, default=DEFAULT_SKIPPED,
                   help=f"Output skip-list JSON (default: {DEFAULT_SKIPPED})")
    p.add_argument("--limit", type=int, default=0,
                   help="Parse only first N galaxies (0 = all)")
    args = p.parse_args()

    galfit_io = import_galfit_skill()

    idx = Table.read(args.index, format="csv")
    eligible = idx[(idx["complexity_rank"] >= 2) & (idx["best_outgal"] != "")]
    if args.limit > 0:
        eligible = eligible[:args.limit]
    print(f"Parsing {len(eligible)} galaxies")

    models: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for i, row in enumerate(eligible, 1):
        name = str(row["name"])
        path = args.mirror / name / str(row["best_outgal"])
        if not path.exists():
            skipped[name] = "missing_local_file"
            continue
        record, skip_reason = parse_one(path, galfit_io)
        if skip_reason:
            skipped[name] = skip_reason
        else:
            models[name] = record
        if i % 100 == 0 or i == len(eligible):
            print(f"  {i:5d}/{len(eligible)}  parsed={len(models):5d}  skipped={len(skipped):4d}")

    args.out_models.parent.mkdir(parents=True, exist_ok=True)
    args.out_models.write_text(json.dumps(models, indent=2))
    args.out_skipped.write_text(json.dumps(skipped, indent=2))
    print(f"\nWrote: {args.out_models}  ({len(models)} models)")
    print(f"Wrote: {args.out_skipped}  ({len(skipped)} skipped)")

    # Skip-reason histogram
    if skipped:
        print("\nSkip reasons:")
        from collections import Counter
        for reason, n in Counter(skipped.values()).most_common():
            print(f"  {reason:30s} {n:5d}")

    # Component-count histogram on the kept models
    if models:
        from collections import Counter
        cc = Counter(m["n_components"] for m in models.values())
        print("\nComponent count distribution (kept models):")
        for n in sorted(cc):
            print(f"  {n} components: {cc[n]:5d}")

        prof_counts = Counter()
        for m in models.values():
            for c in m["components"]:
                prof_counts[c["profile"]] += 1
        print("\nProfile-type frequency (across all components of kept models):")
        for prof, n in prof_counts.most_common():
            print(f"  {prof:12s} {n:6d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
