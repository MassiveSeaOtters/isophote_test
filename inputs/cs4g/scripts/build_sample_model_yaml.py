#!/usr/bin/env python
"""
build_sample_model_yaml.py - Emit a Huang2013-shaped YAML of the sampled galaxies.

Reads `inputs/cs4g/cs4g_components.json` (the full 1198-galaxy mockgal-ready
record set) and `inputs/cs4g/cs4g_sample.csv` (the 198 selected names)
and writes `inputs/cs4g/models/cs4g_sample_models.yaml` in the same
shape mockgal's `load_model_file` consumes for Huang2013.

The emitted model file is configuration-agnostic (kpc sizes + absolute
magnitudes, no pixel or noise info). It is intended to drive mockgal
with whatever run manifest the caller chooses.

Usage:
    python inputs/cs4g/scripts/build_sample_model_yaml.py
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPONENTS = REPO_ROOT / "inputs" / "cs4g" / "cs4g_components.json"
DEFAULT_SAMPLE = REPO_ROOT / "inputs" / "cs4g" / "cs4g_sample.csv"
DEFAULT_OUTPUT = REPO_ROOT / "inputs" / "cs4g" / "models" / "cs4g_sample_models.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    p.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    comps = json.loads(args.components.read_text())
    sample = Table.read(args.sample, format="csv")
    names = [str(r["name"]).strip() for r in sample]
    print(f"Filtering {len(comps)} galaxies down to {len(names)} in sample")

    missing = [n for n in names if n not in comps]
    if missing:
        print(f"WARNING: {len(missing)} sample names missing in components JSON: {missing[:5]}")

    galaxies = []
    for name in names:
        if name not in comps:
            continue
        rec = comps[name]
        galaxies.append({
            "name": name,
            "redshift": float(rec["redshift"]),
            "distance_mpc": float(rec["distance_mpc"]),
            "color_shift_3p6_minus_i": float(rec["color_shift_3p6_minus_i"]),
            # mockgal's load_model_file reads 'components' directly.
            "components": rec["components"],
        })

    document = {
        "metadata": {
            "source": "Salo+2015 S4G P4 GALFIT decompositions (IRSA cs4gcat x Comeron+2025)",
            "description": (
                "Multi-component mockgal models for a stratified 198-galaxy "
                "sub-sample of CS4G late-type disks. Sizes in kpc, magnitudes "
                "in absolute i-band AB (predicted from IRAC1 via Strategy D)."
            ),
            "reference": (
                "Salo+2015 ApJS 219 4; Comeron+2025 A&A 697 A38; "
                "Sheth+2010 PASP 122 1397"
            ),
            "assumed_band": "SDSS i-band AB (predicted)",
            "n_galaxies": len(galaxies),
            "sample_selection": "stratified logMstar, prefer complex; see docs/plan/PLAN_CS4G_SAMPLE.md",
            "component_types": {
                "sersic": "bulge or disk (expdisk converted to Sersic n=1, R_e=1.6783 R_s)",
                "ferrer": "Salo+2015 bar profile; alpha=2, beta=0 typical",
                "psf":    "unresolved galaxy nucleus; requires PSF-enabled render",
            },
        },
        "galaxies": galaxies,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(document, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {args.output}  ({len(galaxies)} galaxies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
