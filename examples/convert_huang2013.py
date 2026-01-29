#!/usr/bin/env python
"""
convert_huang2013.py - Convert Huang 2013 ASCII catalog to YAML/JSON format

This script reads the Huang et al. (2013) CGS model catalog and converts it
to a machine-friendly YAML or JSON format suitable for use with mockgal.py.

Usage:
    python convert_huang2013.py huang2013_cgs_model.txt -o huang2013_models.yaml
    python convert_huang2013.py huang2013_cgs_model.txt -o huang2013_models.json
    python convert_huang2013.py huang2013_cgs_model.txt --galaxy NGC1399 IC1459
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from astropy.cosmology import FlatLambdaCDM

# Default redshift for Huang 2013 galaxies (nearby ellipticals)
HUANG_REDSHIFT = 0.01

# Cosmology for magnitude conversion
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)


def app_to_abs_mag(app_mag: float, z: float) -> float:
    """
    Convert apparent magnitude to absolute magnitude.

    Parameters
    ----------
    app_mag : float
        Apparent magnitude
    z : float
        Redshift

    Returns
    -------
    float
        Absolute magnitude
    """
    dist_mod = float(COSMO.distmod(z).value)
    return app_mag - dist_mod


def parse_huang2013_ascii(filepath: str) -> Dict[str, dict]:
    """
    Parse the Huang 2013 ASCII catalog.

    Parameters
    ----------
    filepath : str
        Path to huang2013_cgs_model.txt

    Returns
    -------
    dict
        Dictionary mapping galaxy names to their data including components
    """
    galaxies = {}

    with open(filepath, 'r') as f:
        for line in f:
            # Skip header and metadata lines
            if line.startswith(('Title', 'Authors', 'Table', '=', '-', 'Byte', 'Note', ' ')):
                continue
            if len(line.strip()) < 30:
                continue

            try:
                # Parse fixed-width columns based on format spec
                # Bytes 1-12: Galaxy name
                name = line[0:12].strip()
                if not name:
                    continue

                # Bytes 14-16: Comment flag
                flag = line[13:16].strip()

                # Bytes 18-25: Model type
                model_type = line[17:25].strip()

                # Bytes 27-31: Reff (kpc)
                reff_str = line[26:31].strip()
                reff = float(reff_str) if reff_str else None

                # Bytes 33-37: Surface brightness (mag/arcsec^2)
                sb_str = line[32:37].strip()
                sb = float(sb_str) if sb_str else None

                # Bytes 39-44: VMag (Absolute V band magnitude - already absolute)
                vmag_str = line[38:44].strip()
                vmag_abs = float(vmag_str) if vmag_str else None

                # Bytes 46-50: Sersic index
                n_str = line[45:50].strip()
                n = float(n_str) if n_str else None

                # Bytes 52-55: Ellipticity
                e_str = line[51:55].strip()
                e = float(e_str) if e_str else 0.0

                # Bytes 57-62: Position angle
                pa_str = line[56:62].strip()
                pa = float(pa_str) if pa_str else 0.0

                # Bytes 64-67: Lopsidedness amplitude
                al_str = line[63:67].strip()
                al = float(al_str) if al_str else None

                # Bytes 69-72: Luminosity fraction
                f_str = line[68:72].strip()
                lum_frac = float(f_str) if f_str else None

                # Skip if essential parameters are missing
                if reff is None or vmag_abs is None or n is None:
                    continue

                # Initialize galaxy entry if new
                if name not in galaxies:
                    galaxies[name] = {
                        'name': name,
                        'flag': flag if flag else None,
                        'redshift': HUANG_REDSHIFT,
                        'components': []
                    }

                # Add component with index
                comp_idx = len(galaxies[name]['components'])
                component = {
                    'id': f"{name}_comp{comp_idx}",
                    'index': comp_idx,
                    'r_eff_kpc': round(reff, 4),
                    'abs_mag': round(vmag_abs, 4),
                    'n': round(n, 4),
                    'ellipticity': round(e, 4),
                    'pa_deg': round(pa, 4),
                }

                # Add optional fields if present
                if sb is not None:
                    component['surface_brightness'] = round(sb, 4)
                if lum_frac is not None:
                    component['lum_fraction'] = round(lum_frac, 4)
                if al is not None:
                    component['lopsidedness'] = round(al, 4)

                galaxies[name]['components'].append(component)

            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue

    return galaxies


def filter_galaxies(
    galaxies: Dict[str, dict],
    names: Optional[List[str]] = None,
    exclude_flags: Optional[List[str]] = None
) -> Dict[str, dict]:
    """
    Filter galaxies by name or exclude by flag.

    Parameters
    ----------
    galaxies : dict
        Full galaxy dictionary
    names : list, optional
        Only include these galaxy names
    exclude_flags : list, optional
        Exclude galaxies with these flags

    Returns
    -------
    dict
        Filtered galaxy dictionary
    """
    result = {}

    for name, data in galaxies.items():
        # Filter by name if specified
        if names is not None and name not in names:
            continue

        # Exclude by flag if specified
        if exclude_flags is not None and data.get('flag') in exclude_flags:
            continue

        result[name] = data

    return result


def convert_to_model_format(galaxies: Dict[str, dict]) -> dict:
    """
    Convert parsed galaxies to the standard Input Model File format.

    Parameters
    ----------
    galaxies : dict
        Parsed galaxy dictionary

    Returns
    -------
    dict
        Model file format with metadata
    """
    # Build galaxy list
    galaxy_list = []
    for name, data in sorted(galaxies.items()):
        galaxy_entry = {
            'name': data['name'],
            'redshift': data['redshift'],
            'components': data['components']
        }
        # Add flag as comment if present
        if data.get('flag'):
            galaxy_entry['flag'] = data['flag']
        galaxy_list.append(galaxy_entry)

    return {
        'metadata': {
            'source': 'Huang et al. (2013) CGS Survey',
            'description': 'Multi-component Sersic models for nearby elliptical galaxies',
            'reference': 'ApJ, 766, 47',
            'assumed_redshift': HUANG_REDSHIFT,
            'n_galaxies': len(galaxy_list),
            'notes': {
                'a': 'Edge-on disk-like structure',
                'b': 'Ring-like structure in outskirts',
                'c': 'Merger remnant',
                'd': 'Contaminated by saturated star',
                'e': 'Significant residuals in model',
                'f': 'Central dust lane',
                'g': 'Possible S0 galaxy'
            }
        },
        'galaxies': galaxy_list
    }


def save_yaml(data: dict, filepath: str) -> None:
    """Save data as YAML file."""
    if not HAS_YAML:
        raise ImportError("PyYAML required for YAML output. Install with: pip install pyyaml")

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def save_json(data: dict, filepath: str) -> None:
    """Save data as JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Huang 2013 ASCII catalog to YAML/JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert entire catalog to YAML
    python convert_huang2013.py huang2013_cgs_model.txt -o huang2013_models.yaml

    # Convert specific galaxies to JSON
    python convert_huang2013.py huang2013_cgs_model.txt -o subset.json --galaxy "NGC 1399" "IC 1459"

    # Exclude problematic galaxies
    python convert_huang2013.py huang2013_cgs_model.txt -o clean.yaml --exclude-flags d e
"""
    )

    parser.add_argument(
        "input",
        help="Path to huang2013_cgs_model.txt"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file path (.yaml or .json)"
    )
    parser.add_argument(
        "--galaxy",
        nargs="+",
        metavar="NAME",
        help="Only include specific galaxies"
    )
    parser.add_argument(
        "--exclude-flags",
        nargs="+",
        metavar="FLAG",
        help="Exclude galaxies with these flags (a-g)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print summary statistics"
    )

    args = parser.parse_args()

    # Parse ASCII catalog
    print(f"Reading: {args.input}")
    galaxies = parse_huang2013_ascii(args.input)
    print(f"Parsed {len(galaxies)} galaxies")

    # Apply filters
    if args.galaxy or args.exclude_flags:
        galaxies = filter_galaxies(galaxies, args.galaxy, args.exclude_flags)
        print(f"After filtering: {len(galaxies)} galaxies")

    # Convert to model format
    model_data = convert_to_model_format(galaxies)

    # Save output
    output_path = Path(args.output)
    if output_path.suffix.lower() in ('.yaml', '.yml'):
        save_yaml(model_data, args.output)
    else:
        save_json(model_data, args.output)

    print(f"Saved: {args.output}")

    # Print summary if verbose
    if args.verbose:
        print("\nSummary:")
        print(f"  Total galaxies: {len(model_data['galaxies'])}")
        n_components = sum(len(g['components']) for g in model_data['galaxies'])
        print(f"  Total components: {n_components}")

        # Component count distribution
        comp_counts = {}
        for g in model_data['galaxies']:
            n = len(g['components'])
            comp_counts[n] = comp_counts.get(n, 0) + 1
        print("  Components per galaxy:")
        for n in sorted(comp_counts.keys()):
            print(f"    {n} components: {comp_counts[n]} galaxies")


if __name__ == "__main__":
    main()
