#!/usr/bin/env python
"""
cs4g_to_mockgal.py - Convert parsed S4G P4 GALFIT models to mockgal manifests.

Reads `inputs/cs4g/cs4g_models.json` (per-galaxy parsed GALFIT decompositions
from `parse_outgals.py`) and `inputs/cs4g/cs4g_candidates.csv` (per-galaxy
distance + predicted i-band absolute magnitude) and emits a MockGalaxy-shaped
JSON record per galaxy.

Per-component conversion rules:

| GALFIT profile  | mockgal component       | col-3 semantics       | size param        |
|-----------------|-------------------------|-----------------------|-------------------|
| sersic          | SersicComponent         | integrated mag        | re_pix            |
| expdisk         | SersicComponent (n=1)   | integrated mag        | rs_pix * 1.6783   |
| devauc          | SersicComponent (n=4)   | integrated mag        | re_pix            |
| gaussian        | SersicComponent (n=0.5) | integrated mag        | fwhm/2.3548       |
| moffat          | SersicComponent (approx)| integrated mag        | fwhm * f(n)       |
| ferrer/ferrer2  | FerrerComponent         | mu(0) [SB at center]  | r_out_pix         |
| psf             | PointSourceComponent    | integrated mag        | -                 |

The Ferrer SB->integrated-magnitude conversion uses the Beta-function form

    F_tot = q * 2*pi * I_0 * r_out_arcsec^2 * I(alpha, beta)
    I(alpha, beta) = 1/(2-beta) * B(beta/(2-beta) + 1, alpha + 1)

so

    m_total = mu(0) - 2.5 * log10(2*pi * q * r_out_arcsec^2 * I(alpha, beta))

Empirically validated: GALFIT ferrer and libprofit ferrer agree to <0.02%
in total flux for typical Salo+2015 bar parameters (alpha=2, beta=0).
GALFIT's `ferrer` and `ferrer2` variants render identically; the
"Surface brightness at FWHM" comment in S4G outgal files is a misleading
annotation, not a real change of normalization.

The per-galaxy magnitude shift uses the `m_i` column from
`cs4g_candidates.csv` (Strategy D / fallback A from D5 evidence). The shift
is applied to every component's absolute magnitude:

    M_i_component = (M_3.6_component) - (M_3.6_galaxy - M_i_galaxy)

i.e., we preserve the per-component flux RATIOS exactly and rigidly shift
the whole galaxy onto the predicted i-band magnitude scale.

Usage:
    python inputs/cs4g/scripts/cs4g_to_mockgal.py
    python inputs/cs4g/scripts/cs4g_to_mockgal.py --limit 5    # smoke run
    python inputs/cs4g/scripts/cs4g_to_mockgal.py --galaxy NGC1097
"""

import argparse
import json
import math
import sys
from pathlib import Path

from astropy.table import Table
from scipy.special import beta as beta_fn


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS = REPO_ROOT / "inputs" / "cs4g" / "cs4g_models.json"
DEFAULT_CANDIDATES = REPO_ROOT / "inputs" / "cs4g" / "cs4g_candidates.csv"
DEFAULT_OUTPUT = REPO_ROOT / "inputs" / "cs4g" / "cs4g_components.json"

# Sersic-equivalent of an exponential disk: R_e = 1.6783 * R_s
EXPDISK_RE_FROM_RS = 1.6783
# Gaussian R_e = sigma * 1.1774 = (FWHM / 2.3548) * 1.1774 = FWHM / 2.0
GAUSSIAN_RE_FROM_FWHM = 0.5

# Plate scale fallback for outgals that don't carry a K) header
DEFAULT_PLATE_SCALE = 0.75   # arcsec / pix (S4G IRAC1 standard)


def ferrer_flux_factor(alpha: float, beta: float) -> float:
    """Compute I(alpha, beta) = (1/(2-beta)) * B(beta/(2-beta)+1, alpha+1).

    The total flux of a circular Ferrer profile is
        F_tot = 2*pi * I_0 * r_out^2 * I(alpha, beta).
    For the common Salo+2015 bar fit (alpha=2, beta=0) this returns 1/6.
    """
    if (2 - beta) <= 0:
        raise ValueError(f"Ferrer (2-beta) must be positive, got beta={beta}")
    return (1.0 / (2.0 - beta)) * beta_fn(beta / (2.0 - beta) + 1.0, alpha + 1.0)


def ferrer_mu0_to_mag(mu0: float, alpha: float, beta: float, q: float,
                      r_out_arcsec: float) -> float:
    """Convert GALFIT Ferrer central SB to the integrated apparent magnitude.

    All inputs in their natural units:
      mu0 in mag/arcsec^2 (the col 3 value)
      q is axis ratio b/a (dimensionless)
      r_out_arcsec is the outer truncation radius in arcsec
      alpha, beta are GALFIT Ferrer shape parameters
    """
    factor = ferrer_flux_factor(alpha, beta)
    if factor <= 0 or r_out_arcsec <= 0 or q <= 0:
        return float("nan")
    return mu0 - 2.5 * math.log10(2.0 * math.pi * q * r_out_arcsec**2 * factor)


def app_to_abs(app_mag: float, distance_mpc: float) -> float:
    """Apparent IRAC1 mag -> 3.6 micron absolute mag using DM."""
    if distance_mpc is None or distance_mpc <= 0 or not math.isfinite(distance_mpc):
        return float("nan")
    return app_mag - (5.0 * math.log10(distance_mpc * 1e6) - 5.0)


def pix_to_arcsec(value_pix: float, plate_scale_arcsec_per_pix: float) -> float:
    return value_pix * plate_scale_arcsec_per_pix


def arcsec_to_kpc(value_arcsec: float, distance_mpc: float) -> float:
    """Small-angle conversion: angle * D_A. Acceptable for z << 1 (true for CS4G)."""
    if distance_mpc is None or distance_mpc <= 0 or not math.isfinite(distance_mpc):
        return float("nan")
    # 1 arcsec = (pi / 648000) rad; D in kpc = distance_mpc * 1e3.
    return value_arcsec * (math.pi / 648000.0) * distance_mpc * 1e3


def _component_to_mockgal(comp: dict, plate_scale: float, distance_mpc: float,
                          color_shift: float) -> dict | None:
    """Map one parsed GALFIT-component dict to a mockgal-component dict.

    color_shift = M_3.6 - M_i (positive value because IRAC1 is brighter).
    Returns None for unhandled profile types (sky, nuker, king, edgedisk).
    """
    profile = comp["profile"]
    suffix = comp.get("flux_norm_suffix")  # e.g. 2 for Salo's ferrer2
    pa_deg = comp.get("pa_deg", 0.0)

    def _abs_mag_from_app(app_mag: float) -> float:
        m_3p6_abs = app_to_abs(app_mag, distance_mpc)
        return m_3p6_abs - color_shift

    # ----- mag-based profiles (col 3 = integrated apparent mag) --------------
    if profile in ("sersic", "devauc", "expdisk", "gaussian", "moffat", "psf"):
        app_mag = comp.get("mag_app")
        if app_mag is None:
            return None
        abs_mag_i = _abs_mag_from_app(app_mag)
    elif profile == "ferrer":
        mu0 = comp.get("mu0", comp.get("mu_central"))
        if mu0 is None:
            return None
        alpha = comp.get("alpha", 2.0)
        beta = comp.get("beta", 0.0)
        q = comp.get("q", 1.0)
        r_out_pix = comp.get("r_out_pix")
        if r_out_pix is None or r_out_pix <= 0:
            return None
        r_out_arcsec = pix_to_arcsec(r_out_pix, plate_scale)
        app_mag = ferrer_mu0_to_mag(mu0, alpha, beta, q, r_out_arcsec)
        if not math.isfinite(app_mag):
            return None
        abs_mag_i = _abs_mag_from_app(app_mag)
    elif profile == "sky":
        return None
    else:
        return None

    if not math.isfinite(abs_mag_i):
        return None

    label = comp.get("structure_label", "")
    component_id = label or f"{profile}_{comp.get('component_number', '')}"

    if profile == "psf":
        return {
            "type": "psf",
            "id": component_id,
            "abs_mag": float(abs_mag_i),
        }

    if profile == "ferrer":
        r_out_arcsec = pix_to_arcsec(r_out_pix, plate_scale)
        r_out_kpc = arcsec_to_kpc(r_out_arcsec, distance_mpc)
        if not math.isfinite(r_out_kpc) or r_out_kpc <= 0:
            return None
        ellip = max(0.0, min(0.999, 1.0 - comp.get("q", 1.0)))
        return {
            "type": "ferrer",
            "id": component_id,
            "r_out_kpc": float(r_out_kpc),
            "abs_mag": float(abs_mag_i),
            "alpha": float(comp.get("alpha", 2.0)),
            "beta": float(comp.get("beta", 0.0)),
            "ellipticity": float(ellip),
            "pa_deg": float(pa_deg),
        }

    # Remaining: sersic-family. Pick the right size + n.
    if profile == "sersic":
        re_pix = comp.get("re_pix")
        n = comp.get("sersic_n", 4.0)
    elif profile == "devauc":
        re_pix = comp.get("re_pix")
        n = 4.0
    elif profile == "expdisk":
        rs_pix = comp.get("rs_pix")
        if rs_pix is None or rs_pix <= 0:
            return None
        re_pix = rs_pix * EXPDISK_RE_FROM_RS
        n = 1.0
    elif profile == "gaussian":
        fwhm_pix = comp.get("fwhm_pix")
        if fwhm_pix is None or fwhm_pix <= 0:
            return None
        re_pix = fwhm_pix * GAUSSIAN_RE_FROM_FWHM
        n = 0.5
    elif profile == "moffat":
        # Moffat -> Sersic is approximate; record FWHM as Re and n=4.7 (Trujillo+2001)
        fwhm_pix = comp.get("fwhm_pix")
        if fwhm_pix is None or fwhm_pix <= 0:
            return None
        re_pix = fwhm_pix * 0.5
        n = 0.5
    else:
        return None

    if re_pix is None or re_pix <= 0:
        return None
    re_arcsec = pix_to_arcsec(re_pix, plate_scale)
    re_kpc = arcsec_to_kpc(re_arcsec, distance_mpc)
    if not math.isfinite(re_kpc) or re_kpc <= 0:
        return None
    ellip = max(0.0, min(0.999, 1.0 - comp.get("q", 1.0)))
    return {
        "type": "sersic",
        "id": component_id,
        "r_eff_kpc": float(re_kpc),
        "abs_mag": float(abs_mag_i),
        "n": float(n),
        "ellipticity": float(ellip),
        "pa_deg": float(pa_deg),
    }


def convert_galaxy(name: str, model: dict, dist_mpc: float, m_i: float,
                   m_3p6_abs: float) -> dict | None:
    """Convert one parsed S4G galaxy dict into a mockgal galaxy dict.

    Returns None if the galaxy has no convertible components after filtering.
    """
    color_shift = m_3p6_abs - m_i  # positive: IRAC1 brighter than i-band
    plate_scale = (model.get("plate_scale_arcsec_per_pix")
                   or DEFAULT_PLATE_SCALE)

    raw_components = model.get("components", [])
    converted = []
    skipped_reasons: dict[str, int] = {}
    for c in raw_components:
        out = _component_to_mockgal(c, plate_scale, dist_mpc, color_shift)
        if out is None:
            reason = c.get("profile", "unknown")
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        else:
            converted.append(out)

    if not converted:
        return None

    # Use a small fixed redshift for the mock-image distance modulus inside
    # mockgal. The CS4G targets are nearby (median ~30 Mpc); the absolute
    # magnitudes we just computed are already correct, so we just need a
    # consistent z for the kpc-to-arcsec conversion when mockgal renders.
    # z = D / (c/H0) ~ D[Mpc] / 4282 (for H0=70). Keep it >= 0.001 to avoid
    # the divide-by-tiny path inside mockgal.
    z_mock = max(dist_mpc / 4282.0, 1e-3)

    return {
        "name": name,
        "redshift": z_mock,
        "distance_mpc": float(dist_mpc),
        "color_shift_3p6_minus_i": float(color_shift),
        "n_components_in": len(raw_components),
        "n_components_out": len(converted),
        "skipped": skipped_reasons or None,
        "components": converted,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--galaxy", type=str, default=None,
                   help="Convert only this galaxy (smoke testing)")
    p.add_argument("--limit", type=int, default=0,
                   help="Convert only first N galaxies (0 = all)")
    args = p.parse_args()

    print(f"Reading {args.models}")
    models = json.loads(args.models.read_text())
    print(f"  {len(models)} parsed S4G models")

    print(f"Reading {args.candidates}")
    cands = Table.read(args.candidates, format="csv")
    by_name = {str(r["name"]).strip(): r for r in cands}
    print(f"  {len(cands)} candidates")

    target_names = list(models.keys())
    if args.galaxy:
        target_names = [args.galaxy]
    if args.limit > 0:
        target_names = target_names[: args.limit]

    out: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for name in target_names:
        if name not in models:
            skipped[name] = "missing_in_models"
            continue
        if name not in by_name:
            skipped[name] = "missing_in_candidates"
            continue
        row = by_name[name]
        dist_mpc = float(row["dist_mpc"])
        m_i = float(row["m_i"])
        m_3p6_abs = float(row["mabs1_3p6"])
        if not (math.isfinite(dist_mpc) and math.isfinite(m_i) and math.isfinite(m_3p6_abs)):
            skipped[name] = "non_finite_catalog_value"
            continue
        rec = convert_galaxy(name, models[name], dist_mpc, m_i, m_3p6_abs)
        if rec is None:
            skipped[name] = "no_convertible_components"
            continue
        out[name] = rec

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {args.output}  ({len(out)} galaxies)")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        from collections import Counter
        for reason, n in Counter(skipped.values()).most_common():
            print(f"  {reason:30s} {n}")

    if out:
        from collections import Counter
        n_in = sum(g["n_components_in"] for g in out.values())
        n_out = sum(g["n_components_out"] for g in out.values())
        prof_in = Counter()
        prof_out = Counter()
        for g in out.values():
            for c in g["components"]:
                prof_out[c["type"]] += 1
            for reason, count in (g.get("skipped") or {}).items():
                prof_in[reason] += count
        print(f"\nComponents: {n_in} in -> {n_out} out")
        if prof_out:
            print("Output mockgal types:")
            for t, n in prof_out.most_common():
                print(f"  {t:10s} {n}")
        if prof_in:
            print("Filtered profile types (per-galaxy):")
            for t, n in prof_in.most_common():
                print(f"  {t:10s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
