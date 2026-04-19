#!/usr/bin/env python
"""
downsample.py - Stratified downsample of the CS4G multi-component sample.

Picks ~N (default 200) galaxies from `cs4g_components.json` such that:

1. Single-component galaxies are dropped (per the "complex models" intent).
2. The selected logMstar distribution closely matches the parent
   multi-component distribution (D4: stratify on logMstar).
3. Within each logMstar bin, galaxies with more components are preferred
   (favors bulge+disk+bar+nucleus over disk+bar), with a deterministic
   PRNG tie-breaker so re-runs are reproducible.

Outputs:
  * `inputs/cs4g/cs4g_sample.csv`  - one row per sampled galaxy
  * `inputs/cs4g/cs4g_sample_summary.json`  - bin-by-bin counts, KS test,
                                              composition breakdown
  * `inputs/cs4g/cs4g_sample_qa.png`  - QA figure: parent vs sample
                                        logMstar / M_i / n_components

Usage:
    python inputs/cs4g/scripts/downsample.py
    python inputs/cs4g/scripts/downsample.py --target-n 100 --min-components 3
    python inputs/cs4g/scripts/downsample.py --bin-width 0.4 --seed 7
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.stats import ks_2samp


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPONENTS = REPO_ROOT / "inputs" / "cs4g" / "cs4g_components.json"
DEFAULT_INDEX = REPO_ROOT / "inputs" / "cs4g" / "cs4g_p4_index.csv"
DEFAULT_CANDIDATES = REPO_ROOT / "inputs" / "cs4g" / "cs4g_candidates.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "inputs" / "cs4g" / "cs4g_sample.csv"
DEFAULT_OUT_JSON = REPO_ROOT / "inputs" / "cs4g" / "cs4g_sample_summary.json"
DEFAULT_OUT_PNG = REPO_ROOT / "inputs" / "cs4g" / "cs4g_sample_qa.png"


def build_parent(components_path: Path, index_path: Path,
                 candidates_path: Path) -> list[dict]:
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
        })
    return parent


def stratified_sample(parent: list[dict], target_n: int,
                      bin_width: float = 0.5, seed: int = 42) -> dict:
    """logMstar-binned proportional sample with n_components tie-break.

    Returns a dict with the sampled subset plus per-bin diagnostics.
    """
    ms = np.array([g["logmstar"] for g in parent])
    if not len(ms):
        return {"sampled": [], "bins": []}

    # Snap edges to the bin_width grid so identical inputs always produce
    # identical bins regardless of which galaxy is at the extremum.
    lo = math.floor(ms.min() / bin_width) * bin_width
    hi = math.ceil(ms.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + bin_width / 2, bin_width)

    rng = np.random.default_rng(seed)
    sampled: list[dict] = []
    bin_records = []
    n_total_parent = len(parent)
    for i in range(len(edges) - 1):
        bin_lo, bin_hi = edges[i], edges[i + 1]
        in_bin = [g for g in parent if bin_lo <= g["logmstar"] < bin_hi]
        # Last bin is closed on the right (catch the maximum)
        if i == len(edges) - 2:
            in_bin = [g for g in parent
                      if bin_lo <= g["logmstar"] <= bin_hi]
        # Proportional target. Round-to-nearest, never round to zero if
        # the bin has any galaxies (preserves tail representation).
        n_target = int(round(target_n * len(in_bin) / n_total_parent))
        if in_bin and n_target == 0 and len(in_bin) >= 1:
            n_target = 1 if (target_n * len(in_bin) / n_total_parent) >= 0.5 else 0

        if not in_bin or n_target == 0:
            bin_records.append({"lo": float(bin_lo), "hi": float(bin_hi),
                                "n_parent": len(in_bin), "n_sampled": 0})
            continue
        if n_target >= len(in_bin):
            sampled.extend(in_bin)
            bin_records.append({"lo": float(bin_lo), "hi": float(bin_hi),
                                "n_parent": len(in_bin),
                                "n_sampled": len(in_bin)})
            continue
        # Sort by n_components desc, then complexity_rank desc, then PRNG
        keyed = sorted(
            in_bin,
            key=lambda g: (
                -g["n_components"], -g["complexity_rank"], rng.random(),
            ),
        )
        sampled.extend(keyed[:n_target])
        bin_records.append({"lo": float(bin_lo), "hi": float(bin_hi),
                            "n_parent": len(in_bin), "n_sampled": n_target})

    return {"sampled": sampled, "bins": bin_records}


def make_qa_figure(parent: list[dict], sampled: list[dict], out_path: Path,
                   bin_width: float) -> None:
    """Three-panel QA: logMstar, M_i, n_components distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    p_ms = np.array([g["logmstar"] for g in parent])
    s_ms = np.array([g["logmstar"] for g in sampled])
    edges = np.arange(p_ms.min(), p_ms.max() + bin_width, bin_width)
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

    n_max = max(max(g["n_components"] for g in parent),
                max(g["n_components"] for g in sampled)) if sampled else 4
    bins_n = np.arange(0.5, n_max + 1.5, 1)
    axes[2].hist([g["n_components"] for g in parent], bins=bins_n,
                 alpha=0.5, density=True, color="grey", label="parent")
    axes[2].hist([g["n_components"] for g in sampled], bins=bins_n,
                 alpha=0.6, density=True, color="crimson", label="sample")
    axes[2].set_xlabel("n_components")
    axes[2].set_ylabel("density")
    axes[2].set_title("n_components per galaxy")
    axes[2].set_xticks(range(1, n_max + 1))
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
    p.add_argument("--target-n", type=int, default=200,
                   help="Target sample size (default 200)")
    p.add_argument("--min-components", type=int, default=2,
                   help="Drop galaxies with fewer than this many components "
                        "(default 2 = drop single-component fits)")
    p.add_argument("--bin-width", type=float, default=0.5,
                   help="logMstar bin width in dex (default 0.5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible tie-breaking")
    args = p.parse_args()

    print(f"Reading {args.components}")
    parent_full = build_parent(args.components, args.index, args.candidates)
    print(f"  {len(parent_full)} galaxies in joined parent")

    parent = [g for g in parent_full if g["n_components"] >= args.min_components]
    n_drop = len(parent_full) - len(parent)
    print(f"  Dropped {n_drop} galaxies with n_components < {args.min_components}")
    print(f"  Eligible parent: {len(parent)} multi-component galaxies")

    if len(parent) <= args.target_n:
        print(f"Parent ({len(parent)}) <= target_n ({args.target_n}); "
              f"keeping all of parent.")
        sampled = list(parent)
        bin_records = []
    else:
        result = stratified_sample(parent, args.target_n,
                                   bin_width=args.bin_width, seed=args.seed)
        sampled = result["sampled"]
        bin_records = result["bins"]

    sampled.sort(key=lambda g: g["name"])

    # KS tests
    p_ms = np.array([g["logmstar"] for g in parent])
    s_ms = np.array([g["logmstar"] for g in sampled])
    p_mi = np.array([g["m_i"] for g in parent])
    s_mi = np.array([g["m_i"] for g in sampled])
    ks_logmstar = ks_2samp(p_ms, s_ms)
    ks_mi = ks_2samp(p_mi, s_mi)

    # Compositions
    from collections import Counter
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
               "m_i", "dist_mpc", "t", "incl_deg", "type", "sample", "mi_source"]
    rows = [{c: g[c] for c in columns} for g in sampled]
    Table(rows=rows, names=columns).write(args.output_csv, format="csv",
                                           overwrite=True)
    print(f"\nSample CSV: {args.output_csv}")

    # Write summary JSON
    summary = {
        "target_n": args.target_n,
        "n_sampled": len(sampled),
        "n_parent_full": len(parent_full),
        "n_parent_eligible": len(parent),
        "min_components": args.min_components,
        "bin_width": args.bin_width,
        "seed": args.seed,
        "ks_logmstar": {"D": float(ks_logmstar.statistic),
                        "p": float(ks_logmstar.pvalue)},
        "ks_m_i": {"D": float(ks_mi.statistic), "p": float(ks_mi.pvalue)},
        "n_components_distribution": dict(n_comp_dist),
        "complexity_rank_distribution": dict(rank_dist),
        "hubble_type_distribution": dict(type_dist),
        "sample_tag_distribution": dict(sample_dist),
        "bin_records": bin_records,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON: {args.output_summary}")

    make_qa_figure(parent, sampled, args.output_qa, args.bin_width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
