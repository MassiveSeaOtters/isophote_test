#!/usr/bin/env python
"""
downsample.py - Size-aware selection of the CS4G multi-component sample.

Builds the production CS4G/S4G mock sample by preferring galaxies that are:

1. Disk-like and non-edge-on already from the candidate build step.
2. Multi-component (default: at least 2 convertible components).
3. Still large enough for 1-D isophotal analysis after being rendered at the
   production HSC-i settings: image size must be > 75 x 75 pixels at z=0.1,
   using the same auto-sizing contract as mockgal.
4. More structurally complex, then larger in rendered apparent size.

The old logMstar-matching objective is retained only as a diagnostic; it is no
longer the driver of the sample choice.

Outputs:
  * `inputs/cs4g/cs4g_sample.csv`  - one row per sampled galaxy
  * `inputs/cs4g/cs4g_sample_summary.json`  - selection counts and diagnostics
  * `inputs/cs4g/cs4g_sample_qa.png`  - QA figure: eligible parent vs sample
                                        in logMstar / M_i / rendered size

Usage:
    python inputs/cs4g/scripts/downsample.py
    python inputs/cs4g/scripts/downsample.py --target-n 300 --min-size-pixels 75
    python inputs/cs4g/scripts/downsample.py --size-redshift 0.1 --size-factor 4
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import ks_2samp


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mockgal import MIN_IMAGE_EXTENT_PIX, kpc_to_arcsec

DEFAULT_COMPONENTS = REPO_ROOT / "inputs" / "cs4g" / "cs4g_components.json"
DEFAULT_INDEX = REPO_ROOT / "inputs" / "cs4g" / "cs4g_p4_index.csv"
DEFAULT_CANDIDATES = REPO_ROOT / "inputs" / "cs4g" / "cs4g_candidates.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "inputs" / "cs4g" / "cs4g_sample.csv"
DEFAULT_OUT_JSON = REPO_ROOT / "inputs" / "cs4g" / "cs4g_sample_summary.json"
DEFAULT_OUT_PNG = REPO_ROOT / "inputs" / "cs4g" / "cs4g_sample_qa.png"


def component_extent_kpc(component: dict) -> float:
    """Return the size-like extent used by mockgal auto-sizing."""
    if "r_eff_kpc" in component:
        return float(component["r_eff_kpc"])
    if "r_out_kpc" in component:
        return float(component["r_out_kpc"])
    return 0.0


def rendered_size_pixels(
    max_component_extent_kpc: float,
    redshift: float,
    pixel_scale: float,
    size_factor: float,
) -> int:
    """Replicate mockgal's auto-size contract for a size-based selection cut."""
    overall_re_pix = kpc_to_arcsec(max_component_extent_kpc, redshift) / pixel_scale
    size = 2 * int(size_factor * overall_re_pix) + 1
    return max(size, MIN_IMAGE_EXTENT_PIX)


def build_parent(
    components_path: Path,
    index_path: Path,
    candidates_path: Path,
) -> list[dict]:
    """Inner-join the three sources into a list of per-galaxy dicts."""
    comps = json.loads(components_path.read_text())
    idx = Table.read(index_path, format="csv")
    cands = Table.read(candidates_path, format="csv")
    idx_by_name = {str(r["name"]).strip(): r for r in idx}
    cand_by_name = {str(r["name"]).strip(): r for r in cands}

    parent = []
    for name, g in comps.items():
        if name not in idx_by_name or name not in cand_by_name:
            continue
        cand = cand_by_name[name]
        max_component_extent_kpc = max(
            (component_extent_kpc(component) for component in g["components"]),
            default=0.0,
        )
        parent.append({
            "name": name,
            "n_components": len(g["components"]),
            "complexity_rank": int(idx_by_name[name]["complexity_rank"]),
            "logmstar": float(cand["logmstar"]),
            "m_i": float(cand["m_i"]),
            "dist_mpc": float(cand["dist_mpc"]),
            "t": float(cand["t"]),
            "incl_deg": float(cand["incl_deg"]),
            "type": str(cand["type"]),
            "sample": str(cand["sample"]),
            "mi_source": str(cand["mi_source"]),
            "max_component_extent_kpc": float(max_component_extent_kpc),
        })
    return parent


def annotate_rendered_sizes(
    galaxies: list[dict],
    redshift: float,
    pixel_scale: float,
    size_factor: float,
) -> list[dict]:
    """Return shallow copies of rows with rendered-size metadata added."""
    annotated = []
    for galaxy in galaxies:
        row = dict(galaxy)
        row["size_pixels_cut"] = rendered_size_pixels(
            max_component_extent_kpc=row["max_component_extent_kpc"],
            redshift=redshift,
            pixel_scale=pixel_scale,
            size_factor=size_factor,
        )
        annotated.append(row)
    return annotated


def select_sample(parent: list[dict], target_n: int) -> list[dict]:
    """Prefer structurally richer and larger galaxies deterministically."""
    ranked = sorted(
        parent,
        key=lambda galaxy: (
            -galaxy["complexity_rank"],
            -galaxy["n_components"],
            -galaxy["size_pixels_cut"],
            -galaxy["logmstar"],
            galaxy["name"],
        ),
    )
    return ranked[:target_n]


def make_qa_figure(parent: list[dict], sampled: list[dict], out_path: Path) -> None:
    """Three-panel QA: logMstar, M_i, rendered size distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    p_ms = np.array([g["logmstar"] for g in parent])
    s_ms = np.array([g["logmstar"] for g in sampled])
    edges = np.arange(p_ms.min(), p_ms.max() + 0.5, 0.5)
    axes[0].hist(p_ms, bins=edges, alpha=0.5, density=True,
                 color="grey", label=f"parent (N={len(p_ms)})")
    axes[0].hist(s_ms, bins=edges, alpha=0.6, density=True,
                 color="crimson", label=f"sample (N={len(s_ms)})")
    axes[0].set_xlabel(r"$\log M_\star / M_\odot$")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"logMstar (KS={ks_2samp(p_ms, s_ms).statistic:.3f})")
    axes[0].legend()

    p_mi = np.array([g["m_i"] for g in parent])
    s_mi = np.array([g["m_i"] for g in sampled])
    edges_mi = np.arange(p_mi.min(), p_mi.max() + 0.5, 0.5)
    axes[1].hist(p_mi, bins=edges_mi, alpha=0.5, density=True,
                 color="grey", label="parent")
    axes[1].hist(s_mi, bins=edges_mi, alpha=0.6, density=True,
                 color="crimson", label="sample")
    axes[1].set_xlabel(r"predicted $M_i$")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"M_i (KS={ks_2samp(p_mi, s_mi).statistic:.3f})")
    axes[1].invert_xaxis()
    axes[1].legend()

    p_size = np.array([g["size_pixels_cut"] for g in parent])
    s_size = np.array([g["size_pixels_cut"] for g in sampled])
    size_edges = np.arange(p_size.min(), p_size.max() + 20, 20)
    axes[2].hist(p_size, bins=size_edges,
                 alpha=0.5, density=True, color="grey", label="parent")
    axes[2].hist(s_size, bins=size_edges,
                 alpha=0.6, density=True, color="crimson", label="sample")
    axes[2].set_xlabel("rendered size at z=0.1 (pixels)")
    axes[2].set_ylabel("density")
    axes[2].set_title("size-aware eligibility")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"QA figure: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--output-summary", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--output-qa", type=Path, default=DEFAULT_OUT_PNG)
    p.add_argument("--target-n", type=int, default=300,
                   help="Target sample size (default 300)")
    p.add_argument("--min-components", type=int, default=2,
                   help="Drop galaxies with fewer than this many components "
                        "(default 2 = drop single-component fits)")
    p.add_argument("--size-redshift", type=float, default=0.1,
                   help="Redshift used for the apparent-size eligibility cut (default 0.1)")
    p.add_argument("--pixel-scale", type=float, default=0.168,
                   help="Pixel scale used for the apparent-size cut (default 0.168)")
    p.add_argument("--size-factor", type=float, default=4.0,
                   help="Auto-size factor used for the apparent-size cut (default 4.0)")
    p.add_argument("--min-size-pixels", type=int, default=75,
                   help="Require rendered size to be strictly larger than this many pixels "
                        "(default 75)")
    args = p.parse_args()

    print(f"Reading {args.components}")
    parent_full = build_parent(args.components, args.index, args.candidates)
    print(f"  {len(parent_full)} galaxies in joined parent")

    parent_components = [g for g in parent_full if g["n_components"] >= args.min_components]
    n_drop = len(parent_full) - len(parent_components)
    print(f"  Dropped {n_drop} galaxies with n_components < {args.min_components}")

    parent_annotated = annotate_rendered_sizes(
        parent_components,
        redshift=args.size_redshift,
        pixel_scale=args.pixel_scale,
        size_factor=args.size_factor,
    )
    parent = [
        galaxy for galaxy in parent_annotated
        if galaxy["size_pixels_cut"] > args.min_size_pixels
    ]
    print(
        f"  Eligible parent after size cut: {len(parent)} galaxies "
        f"(size_pixels > {args.min_size_pixels} at z={args.size_redshift})"
    )

    if len(parent) <= args.target_n:
        print(f"Parent ({len(parent)}) <= target_n ({args.target_n}); "
              f"keeping all of parent.")
        sampled = list(parent)
    else:
        sampled = select_sample(parent, args.target_n)

    sampled.sort(key=lambda g: g["name"])

    # KS tests
    p_ms = np.array([g["logmstar"] for g in parent])
    s_ms = np.array([g["logmstar"] for g in sampled])
    p_mi = np.array([g["m_i"] for g in parent])
    s_mi = np.array([g["m_i"] for g in sampled])
    ks_logmstar = ks_2samp(p_ms, s_ms)
    ks_mi = ks_2samp(p_mi, s_mi)

    # Compositions
    n_comp_dist = Counter(g["n_components"] for g in sampled)
    rank_dist = Counter(g["complexity_rank"] for g in sampled)
    type_dist = Counter(g["type"].strip() for g in sampled)
    sample_dist = Counter(g["sample"].strip() for g in sampled)

    print(f"\nSample size: {len(sampled)} (target {args.target_n})")
    print(f"  KS(logMstar): D={ks_logmstar.statistic:.4f}  p={ks_logmstar.pvalue:.3f}")
    print(f"  KS(M_i):      D={ks_mi.statistic:.4f}  p={ks_mi.pvalue:.3f}")
    print(f"\nn_components in sample:")
    for n in sorted(n_comp_dist):
        print(f"  {n}-comp: {n_comp_dist[n]:4d}")
    print(f"\nComplexity rank in sample:")
    for r in sorted(rank_dist):
        print(f"  rank {r}: {rank_dist[r]:4d}")
    print(f"\nTop 8 Hubble types in sample:")
    for t, n in type_dist.most_common(8):
        print(f"  {t:12s} {n:4d}")
    print(f"\nSub-sample tags:")
    for s, n in sample_dist.most_common():
        print(f"  {s:8s} {n:4d}")

    # Write CSV
    columns = ["name", "complexity_rank", "n_components", "logmstar",
               "m_i", "dist_mpc", "t", "incl_deg", "type", "sample", "mi_source",
               "max_component_extent_kpc", "size_pixels_cut"]
    rows = [{c: g[c] for c in columns} for g in sampled]
    Table(rows=rows, names=columns).write(args.output_csv, format="csv",
                                           overwrite=True)
    print(f"\nSample CSV: {args.output_csv}")

    # Write summary JSON
    summary = {
        "target_n": args.target_n,
        "n_sampled": len(sampled),
        "n_parent_full": len(parent_full),
        "n_parent_after_component_cut": len(parent_components),
        "n_parent_eligible": len(parent),
        "min_components": args.min_components,
        "size_cut": {
            "redshift": args.size_redshift,
            "pixel_scale": args.pixel_scale,
            "size_factor": args.size_factor,
            "min_size_pixels_exclusive": args.min_size_pixels,
        },
        "ks_logmstar": {"D": float(ks_logmstar.statistic),
                        "p": float(ks_logmstar.pvalue)},
        "ks_m_i": {"D": float(ks_mi.statistic), "p": float(ks_mi.pvalue)},
        "n_components_distribution": dict(n_comp_dist),
        "complexity_rank_distribution": dict(rank_dist),
        "hubble_type_distribution": dict(type_dist),
        "sample_tag_distribution": dict(sample_dist),
        "size_pixels_cut_range": {
            "min": int(min(g["size_pixels_cut"] for g in sampled)) if sampled else None,
            "median": float(np.median([g["size_pixels_cut"] for g in sampled])) if sampled else None,
            "max": int(max(g["size_pixels_cut"] for g in sampled)) if sampled else None,
        },
    }
    args.output_summary.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON: {args.output_summary}")

    make_qa_figure(parent, sampled, args.output_qa)
    return 0


if __name__ == "__main__":
    sys.exit(main())
