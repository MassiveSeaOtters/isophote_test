#!/usr/bin/env python
"""
build_sample.py - Build the CS4G candidate sample for mock-galaxy generation.

Cross-joins the IRSA cs4gcat catalog with the VizieR J/A+A/697/A38 (Comeron+2025)
catalog, applies late-type + non-edge-on cuts, and computes a predicted i-band
absolute magnitude per galaxy from IRAC1 + B-band photometry. The 366 dwarf
galaxies that have a real SDSS i-band cross-match (`mabs3`) carry the real
value; everyone else gets a predicted M_i tagged with its provenance.

See `docs/plan/PLAN_CS4G_SAMPLE.md` (decisions D1, D2, D5, D6) for rationale.

Usage:
    python inputs/cs4g/scripts/build_sample.py
    python inputs/cs4g/scripts/build_sample.py --t-cut -3 --incl-cut 70 -o my_candidates.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IRSA = REPO_ROOT / "inputs" / "cs4g" / "cs4g_catalog.fits"
DEFAULT_VIZ = REPO_ROOT / "inputs" / "cs4g" / "cs4g_2025.fit"
DEFAULT_OUT = REPO_ROOT / "inputs" / "cs4g" / "cs4g_candidates.csv"

# M_i prediction calibration, fit on the 366 dwarf-galaxy training set
# (decisions D5/D6, evidence in `.scratch/cs4g_d4_d5_evidence.png`).
COLOR_CONST = 0.573      # M_3.6 - M_i, median of the 366 calibrators
COLOR_B_SLOPE = -0.235   # color = SLOPE * (B_abs - M_3.6) + INTERCEPT
COLOR_B_INTERCEPT = 0.670
# Linear fit was calibrated on (B_abs - M_3.6) in roughly [-1, 5]; outside
# this we fall back to the constant offset rather than extrapolate.
B_IRAC_RANGE = (-2.0, 6.0)


def predict_m_i(m_3p6_abs: np.ndarray, b_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict M_i from M_3.6 and (optionally) absolute B-band magnitude.

    For each galaxy, returns (m_i_predicted, source_tag) where source_tag is
    'predicted_B' if Strategy D applied, else 'predicted_const'.
    """
    n = len(m_3p6_abs)
    pred = np.full(n, np.nan)
    src = np.full(n, "predicted_const", dtype="<U16")

    has_b = np.isfinite(b_abs)
    color_b = b_abs - m_3p6_abs
    in_range = has_b & (color_b > B_IRAC_RANGE[0]) & (color_b < B_IRAC_RANGE[1])

    color = COLOR_B_SLOPE * color_b + COLOR_B_INTERCEPT
    pred = np.where(in_range, m_3p6_abs - color, m_3p6_abs - COLOR_CONST)
    src = np.where(in_range, "predicted_B", "predicted_const")
    return pred, src


def build_candidates(irsa_path: Path, viz_path: Path,
                     t_cut: float, incl_cut: float) -> Table:
    """Cross-join the two catalogs and apply the candidate cuts."""
    irsa = Table.read(irsa_path, hdu=1)
    viz = Table.read(viz_path, hdu=1)

    irsa_obj = np.array([str(x).strip() for x in irsa["object"]])
    viz_obj = np.array([str(x).strip() for x in viz["Object"]])
    viz_idx = {o: i for i, o in enumerate(viz_obj)}

    # Pull Incl and Btot from the Vizier rows aligned to the IRSA index.
    incl = np.full(len(irsa), np.nan)
    btot_app = np.full(len(irsa), np.nan)
    for i, o in enumerate(irsa_obj):
        j = viz_idx.get(o)
        if j is None:
            continue
        incl[i] = float(viz["Incl"][j])
        v = float(viz["Btot"][j])
        if np.isfinite(v) and v > 0:
            btot_app[i] = v

    t_type = np.asarray(irsa["cvrhs_t"], dtype=float)
    mabs1 = np.asarray(irsa["mabs1"], dtype=float)     # M_3.6 absolute
    mabs3 = np.asarray(irsa["mabs3"], dtype=float)     # real M_i (DG only)
    mstar = np.asarray(irsa["mstar"], dtype=float)     # log10(Mstar/Msun)
    dist = np.asarray(irsa["distance"], dtype=float)   # Mpc

    # Distance modulus and absolute B
    dm = np.where(np.isfinite(dist) & (dist > 0),
                  5 * np.log10(dist * 1e6) - 5,
                  np.nan)
    b_abs = btot_app - dm

    # Apply the candidate cuts (D1, D2, plus IRAC1 + Mstar finite for downsampling)
    mask = (
        (t_type >= t_cut)
        & np.isfinite(t_type)
        & (incl < incl_cut)
        & np.isfinite(incl)
        & np.isfinite(mabs1)
        & np.isfinite(mstar)
        & np.isfinite(dist)
    )

    # Predict M_i for everyone in the candidate sample.
    m_i_pred, mi_src = predict_m_i(mabs1[mask], b_abs[mask])

    # Real mabs3 wins where present (DG sub-sample only).
    real_mi = mabs3[mask]
    has_real = np.isfinite(real_mi)
    m_i_final = np.where(has_real, real_mi, m_i_pred)
    mi_src = np.where(has_real, "real", mi_src)

    out = Table()
    out["name"] = np.asarray(irsa["object"], dtype=str)[mask]
    out["ra"] = np.asarray(irsa["ra"])[mask]
    out["dec"] = np.asarray(irsa["dec"])[mask]
    out["sample"] = np.asarray(irsa["sample"], dtype=str)[mask]
    out["type"] = np.asarray(irsa["type"], dtype=str)[mask]
    out["cvrhs"] = np.asarray(irsa["cvrhs"], dtype=str)[mask]
    out["t"] = t_type[mask]
    out["incl_deg"] = incl[mask]
    out["dist_mpc"] = dist[mask]
    out["distmod"] = dm[mask]
    out["mabs1_3p6"] = mabs1[mask]
    out["logmstar"] = mstar[mask]
    out["btot_app"] = btot_app[mask]
    out["b_abs"] = b_abs[mask]
    out["m_i"] = m_i_final
    out["mi_source"] = mi_src
    out["mabs3_real"] = real_mi
    out["p4_url"] = [
        f"https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{n}/P4/"
        for n in out["name"]
    ]

    return out


def summarize(tbl: Table) -> None:
    print(f"Candidate sample: {len(tbl)} galaxies")
    print(f"  Sample tags:  ", dict(zip(*np.unique(tbl["sample"], return_counts=True))))
    print(f"  M_i sources:  ", dict(zip(*np.unique(tbl["mi_source"], return_counts=True))))
    print(f"  T-type range: [{tbl['t'].min():.1f}, {tbl['t'].max():.1f}]")
    print(f"  Incl range:   [{tbl['incl_deg'].min():.1f}, {tbl['incl_deg'].max():.1f}] deg")
    print(f"  Dist range:   [{tbl['dist_mpc'].min():.1f}, {tbl['dist_mpc'].max():.1f}] Mpc")
    print(f"  M_i range:    [{tbl['m_i'].min():.2f}, {tbl['m_i'].max():.2f}]")
    print(f"  logMstar:     [{tbl['logmstar'].min():.2f}, {tbl['logmstar'].max():.2f}]")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--irsa", type=Path, default=DEFAULT_IRSA,
                   help=f"IRSA cs4gcat FITS (default: {DEFAULT_IRSA})")
    p.add_argument("--viz", type=Path, default=DEFAULT_VIZ,
                   help=f"VizieR J/A+A/697/A38 FITS (default: {DEFAULT_VIZ})")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                   help=f"Output CSV (default: {DEFAULT_OUT})")
    p.add_argument("--t-cut", type=float, default=-3.0,
                   help="Minimum cvrhs_t T-type (default: -3, S0- onwards)")
    p.add_argument("--incl-cut", type=float, default=70.0,
                   help="Maximum inclination in degrees (default: 70)")
    args = p.parse_args()

    tbl = build_candidates(args.irsa, args.viz, args.t_cut, args.incl_cut)
    summarize(tbl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tbl.write(args.output, format="csv", overwrite=True)
    print(f"\nWrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
