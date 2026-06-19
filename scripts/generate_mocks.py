#!/usr/bin/env python
"""
Generate systematic mock images for any mockgal-compatible sample from a run
manifest. Dataset-agnostic: the manifest's `model_file` decides which sample
(Huang2013, CS4G/s4g_mock, etc.) is rendered; the `--output` flag is the
dataset's root folder. Per-galaxy FITS files land at
`{output}/{galaxy}/{galaxy}_{config}.fits`, alongside a `{galaxy}_mosaic.png`
QA mosaic. Root-level `run_manifest_original.yaml`, `run_manifest_resolved.yaml`,
and `run_metadata.json` record the run for reproducibility.

Usage:
    python scripts/generate_mocks.py \
        --run-manifest inputs/huang2013/runs/huang2013_hsc_i_wide.yaml \
        --output ~/Dropbox/work/data/huang2013

    python scripts/generate_mocks.py \
        --run-manifest inputs/cs4g/runs/cs4g_hsc_i_wide.yaml \
        --output ~/Dropbox/work/data/s4g_mock

    python scripts/generate_mocks.py \
        --run-manifest inputs/cs4g/runs/cs4g_hsc_i_wide.yaml \
        --output ~/Dropbox/work/data/s4g_mock \
        --galaxies NGC1433 NGC0275 NGC1357 \
        --config-name clean_z005 wide_z020
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mockgal import ImageConfig, MockGalaxy, MockImageGenerator, load_model_file, sanitize_filename, save_fits

SCRIPT_DEFAULTS: dict[str, Any] = {
    "pixel_scale": 0.168,
    "zeropoint": 27.0,
    "size_factor": 6.0,
    "size_pixels": None,
    "max_image_size": 4001,
    "psf_enabled": True,
    "psf_type": "gaussian",
    "psf_fwhm": 0.7,
    "psf_moffat_beta": 4.765,
    "psf_file": None,
    "sky_enabled": False,
    "sky_type": "flat",
    "sky_level": 0.0,
    "sky_coeffs": [0.0],
    "sky_sb_value": None,
    "noise_enabled": False,
    "noise_sigma": None,
    "noise_snr": None,
    "noise_seed": 42,
    "sky_sb_limit": None,
    "gain": None,
    "randomize_noise_seed": False,
    "engine": "auto",
    "profit_cli_path": None,
}

ALLOWED_TOP_LEVEL_KEYS = {
    "run_name",
    "description",
    "model_file",
    "output_root",
    "defaults",
    "configs",
}
ALLOWED_ROW_KEYS = {"name", "redshift", *SCRIPT_DEFAULTS.keys()}
REQUIRED_ROW_KEYS = {"name", "redshift", "size_factor"}
NOISE_DRIVER_KEYS = ("noise_sigma", "noise_snr", "sky_sb_limit", "sky_sb_value")


@dataclass(frozen=True, slots=True)
class ResolvedRunRow:
    name: str
    redshift: float
    config_values: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedRunManifest:
    run_name: str
    description: str | None
    model_file: Path
    output_root: Path | None   # informational only; actual dataset root is --output
    defaults: dict[str, Any]
    rows: list[ResolvedRunRow]
    source_manifest: Path


def load_yaml_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def validate_row_keys(row: dict[str, Any], row_index: int) -> None:
    unknown_keys = sorted(set(row) - ALLOWED_ROW_KEYS)
    if unknown_keys:
        raise ValueError(f"Config row {row_index} has unknown keys: {', '.join(unknown_keys)}")

    missing_keys = [key for key in REQUIRED_ROW_KEYS if key not in row]
    if missing_keys:
        raise ValueError(
            f"Config row {row_index} is missing required keys: {', '.join(sorted(missing_keys))}"
        )


def validate_defaults(defaults: dict[str, Any]) -> None:
    unknown_keys = sorted(set(defaults) - set(SCRIPT_DEFAULTS))
    if unknown_keys:
        raise ValueError(f"Manifest defaults have unknown keys: {', '.join(unknown_keys)}")


def validate_noise_configuration(row_name: str, config_values: dict[str, Any]) -> None:
    if not config_values.get("noise_enabled", False):
        return
    if all(config_values.get(key) is None for key in NOISE_DRIVER_KEYS):
        raise ValueError(
            f"Config '{row_name}' sets noise_enabled=true but does not provide any "
            "noise driver (noise_sigma, noise_snr, sky_sb_limit, or sky_sb_value)"
        )


def build_image_config(config_name: str, config_values: dict[str, Any]) -> ImageConfig:
    return ImageConfig(
        name=config_name,
        pixel_scale=config_values["pixel_scale"],
        zeropoint=config_values["zeropoint"],
        size_factor=config_values["size_factor"],
        size_pixels=config_values["size_pixels"],
        max_image_size=config_values["max_image_size"],
        psf_enabled=config_values["psf_enabled"],
        psf_type=config_values["psf_type"],
        psf_fwhm=config_values["psf_fwhm"],
        psf_moffat_beta=config_values["psf_moffat_beta"],
        psf_file=config_values["psf_file"],
        sky_enabled=config_values["sky_enabled"],
        sky_type=config_values["sky_type"],
        sky_level=config_values["sky_level"],
        sky_coeffs=config_values["sky_coeffs"],
        sky_sb_value=config_values["sky_sb_value"],
        noise_enabled=config_values["noise_enabled"],
        noise_sigma=config_values["noise_sigma"],
        noise_snr=config_values["noise_snr"],
        noise_seed=config_values["noise_seed"],
        sky_sb_limit=config_values["sky_sb_limit"],
        gain=config_values["gain"],
        randomize_noise_seed=config_values["randomize_noise_seed"],
        engine=config_values["engine"],
        profit_cli_path=config_values["profit_cli_path"],
    )


def resolve_run_manifest(manifest_path: Path) -> ResolvedRunManifest:
    raw_data = load_yaml_file(manifest_path)
    if raw_data is None:
        raise ValueError(f"Manifest {manifest_path} is empty")

    manifest = require_mapping(raw_data, "Run manifest")
    unknown_top_level = sorted(set(manifest) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise ValueError(
            f"Run manifest {manifest_path} has unknown top-level keys: "
            f"{', '.join(unknown_top_level)}"
        )

    for required_key in ("run_name", "model_file", "configs"):
        if required_key not in manifest:
            raise ValueError(f"Run manifest {manifest_path} is missing required key '{required_key}'")

    defaults = dict(SCRIPT_DEFAULTS)
    manifest_defaults = manifest.get("defaults", {})
    manifest_defaults = require_mapping(manifest_defaults, "Manifest defaults")
    validate_defaults(manifest_defaults)
    defaults.update(manifest_defaults)

    model_file = Path(manifest["model_file"])
    # `output_root` in the manifest is informational only (recorded in metadata);
    # the actual dataset root comes from --output on the CLI.
    output_root_value = manifest.get("output_root")
    output_root = Path(output_root_value) if output_root_value else None
    rows: list[ResolvedRunRow] = []
    seen_names: set[str] = set()

    for row_index, row_value in enumerate(require_list(manifest["configs"], "Manifest configs"), start=1):
        row = require_mapping(row_value, f"Config row {row_index}")
        validate_row_keys(row, row_index)

        config_name = row.get("name")
        if not config_name or not isinstance(config_name, str):
            raise ValueError(f"Config row {row_index} must include a non-empty string name")
        if config_name in seen_names:
            raise ValueError(f"Duplicate config name '{config_name}' in manifest {manifest_path}")
        seen_names.add(config_name)

        redshift = row.get("redshift")
        if redshift is None:
            raise ValueError(f"Config '{config_name}' is missing required key 'redshift'")
        if not isinstance(redshift, (int, float)) or redshift <= 0:
            raise ValueError(f"Config '{config_name}' must define a positive redshift")

        config_values = dict(defaults)
        for key in SCRIPT_DEFAULTS:
            if key in row:
                config_values[key] = row[key]

        validate_noise_configuration(config_name, config_values)
        build_image_config(config_name, config_values)
        rows.append(
            ResolvedRunRow(
                name=config_name,
                redshift=float(redshift),
                config_values=config_values,
            )
        )

    return ResolvedRunManifest(
        run_name=str(manifest["run_name"]),
        description=manifest.get("description"),
        model_file=model_file,
        output_root=output_root,
        defaults=defaults,
        rows=rows,
        source_manifest=manifest_path,
    )


def select_rows(
    rows: list[ResolvedRunRow],
    requested_names: list[str] | None,
) -> list[ResolvedRunRow]:
    if not requested_names:
        return rows

    requested_set = set(requested_names)
    available_names = {row.name for row in rows}
    missing_names = sorted(requested_set - available_names)
    if missing_names:
        raise ValueError(
            "Unknown config names requested: "
            + ", ".join(missing_names)
        )
    return [row for row in rows if row.name in requested_set]


def get_git_branch() -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    branch = result.stdout.strip()
    return branch or None


def copy_original_manifest(source_path: Path, output_base: Path) -> Path:
    destination = output_base / "run_manifest_original.yaml"
    shutil.copy2(source_path, destination)
    return destination


def serialize_resolved_manifest(manifest: ResolvedRunManifest, rows: list[ResolvedRunRow]) -> dict[str, Any]:
    return {
        "run_name": manifest.run_name,
        "description": manifest.description,
        "model_file": str(manifest.model_file),
        "output_root": str(manifest.output_root) if manifest.output_root else None,
        "defaults": manifest.defaults,
        "configs": [
            {
                "name": row.name,
                "redshift": row.redshift,
                **row.config_values,
            }
            for row in rows
        ],
    }


def write_resolved_manifest(
    output_base: Path,
    manifest: ResolvedRunManifest,
    rows: list[ResolvedRunRow],
) -> Path:
    destination = output_base / "run_manifest_resolved.yaml"
    payload = serialize_resolved_manifest(manifest, rows)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def write_run_metadata(
    output_base: Path,
    manifest: ResolvedRunManifest,
    rows: list[ResolvedRunRow],
    galaxies: list[MockGalaxy],
    selected_galaxy_names: list[str] | None,
    selected_config_names: list[str] | None,
    warnings: list[str],
) -> Path:
    metadata_path = output_base / "run_metadata.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "git_branch": get_git_branch(),
        "source_manifest": str(manifest.source_manifest),
        "run_name": manifest.run_name,
        "description": manifest.description,
        "model_file": str(manifest.model_file),
        "output_root": str(manifest.output_root) if manifest.output_root else None,
        "selected_config_names": selected_config_names,
        "selected_galaxy_names": selected_galaxy_names,
        "galaxy_count": len(galaxies),
        "config_count": len(rows),
        "expected_image_count": len(galaxies) * len(rows),
        "warnings": warnings,
        "resolved_rows": [
            {
                "name": row.name,
                "redshift": row.redshift,
                **row.config_values,
            }
            for row in rows
        ],
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def format_row_label(row: ResolvedRunRow) -> str:
    parts = [f"{row.name}", f"z={row.redshift:.2f}"]
    if row.config_values.get("noise_enabled"):
        if row.config_values.get("sky_sb_limit") is not None:
            parts.append(f"sb_lim={row.config_values['sky_sb_limit']}")
        elif row.config_values.get("sky_sb_value") is not None:
            parts.append(f"sb_val={row.config_values['sky_sb_value']}")
        elif row.config_values.get("noise_sigma") is not None:
            parts.append(f"sigma={row.config_values['noise_sigma']}")
        elif row.config_values.get("noise_snr") is not None:
            parts.append(f"snr={row.config_values['noise_snr']}")
        else:
            parts.append("noise")
    else:
        parts.append("no noise")
    return ", ".join(parts)


def create_mosaic(
    galaxy_name: str,
    galaxy_dir: Path,
    rows: list[ResolvedRunRow],
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.io import fits
    except ImportError:
        print("  Warning: matplotlib or astropy not available, skipping mosaic")
        return None

    count = len(rows)
    if count == 0:
        return None

    columns = 2 if count <= 4 else 3
    rows_count = math.ceil(count / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(6 * columns, 6 * rows_count))
    axes_array = np.atleast_1d(axes).ravel()
    fig.suptitle(galaxy_name, fontsize=16, fontweight="bold")

    for axis in axes_array[count:]:
        axis.axis("off")

    for row, axis in zip(rows, axes_array):
        fits_path = galaxy_dir / f"{galaxy_name}_{row.name}.fits"
        if not fits_path.exists():
            axis.set_title(f"{format_row_label(row)} (missing)", fontsize=10)
            axis.axis("off")
            continue

        with fits.open(fits_path) as hdul:
            image = hdul[0].data

        scale = np.nanpercentile(image, 99.5) / 10.0
        if scale <= 0:
            scale = 1.0
        scaled = np.arcsinh(image / scale)
        axis.imshow(scaled, origin="lower", cmap="magma", interpolation="nearest")
        axis.set_title(format_row_label(row), fontsize=10)
        axis.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = galaxy_dir / f"{galaxy_name}_mosaic.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def process_galaxy(
    galaxy: MockGalaxy,
    rows: list[ResolvedRunRow],
    output_base: Path,
) -> None:
    safe_galaxy_name = sanitize_filename(galaxy.name)
    galaxy_dir = output_base / safe_galaxy_name
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {galaxy.name}...")

    for index, row in enumerate(rows, start=1):
        galaxy_at_z = MockGalaxy(
            name=galaxy.name,
            redshift=row.redshift,
            components=galaxy.components,
            re_overall=galaxy.re_overall,
        )
        config = build_image_config(row.name, row.config_values)
        generator = MockImageGenerator(config)
        image, metadata = generator.generate(galaxy_at_z)
        output_path = galaxy_dir / f"{safe_galaxy_name}_{row.name}.fits"
        save_fits(image, metadata, output_path)
        print(
            f"  [{index}/{len(rows)}] {row.name}: z={row.redshift:.2f}, "
            f"size={image.shape}, saved to {output_path.name}"
        )

    mosaic_path = create_mosaic(safe_galaxy_name, galaxy_dir, rows)
    if mosaic_path:
        print(f"  Mosaic saved to {mosaic_path.name}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate systematic mock galaxy images from a run manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        required=True,
        help="YAML manifest that defines the production run rows",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for generated mocks and run artifacts",
    )
    parser.add_argument(
        "--galaxies",
        nargs="+",
        type=str,
        default=None,
        help="Process only specific galaxies from the model file",
    )
    parser.add_argument(
        "--config-name",
        nargs="+",
        type=str,
        default=None,
        help="Process only specific manifest configuration names",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only the first 2 galaxies after selection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest and print the resolved run without generating files",
    )
    return parser


def load_galaxies(model_file: Path, galaxy_names: list[str] | None, test_mode: bool) -> list[MockGalaxy]:
    galaxies = load_model_file(str(model_file))
    if galaxy_names:
        requested = set(galaxy_names)
        galaxies = [galaxy for galaxy in galaxies if galaxy.name in requested]
        print(f"Processing {len(galaxies)} selected galaxies: {', '.join(galaxy_names)}")
    if test_mode:
        galaxies = galaxies[:2]
        if galaxies:
            selected_names = ", ".join(galaxy.name for galaxy in galaxies)
            print(f"Test mode: Processing first {len(galaxies)} galaxies: {selected_names}")
    return galaxies


def main() -> int:
    args = create_parser().parse_args()
    manifest = resolve_run_manifest(args.run_manifest)
    rows = select_rows(manifest.rows, args.config_name)
    if not rows:
        raise ValueError("No manifest configuration rows selected")

    warnings: list[str] = []
    for row in rows:
        if row.config_values.get("sky_sb_value") is not None and row.config_values.get("sky_sb_limit") is not None:
            warnings.append(
                f"Config '{row.name}' defines both sky_sb_value and sky_sb_limit; "
                "MockGal will prefer sky_sb_value."
            )

    print(f"Loading galaxy models from {manifest.model_file}")
    galaxies = load_galaxies(manifest.model_file, args.galaxies, args.test)
    print(f"Loaded {len(galaxies)} galaxies after selection")
    if not galaxies:
        print("Error: No galaxies to process!")
        return 1

    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Run manifest: {manifest.source_manifest}")
    print(f"Run name: {manifest.run_name}")
    print(f"Output directory: {args.output}")
    print(f"Number of galaxies: {len(galaxies)}")
    print(f"Number of configs: {len(rows)}")
    print(f"Expected images: {len(galaxies) * len(rows)}")
    print("Configs:")
    for row in rows:
        print(f"  - {format_row_label(row)}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("=" * 70)

    if args.dry_run:
        return 0

    output_base = args.output
    output_base.mkdir(parents=True, exist_ok=True)
    original_manifest_path = copy_original_manifest(manifest.source_manifest, output_base)
    resolved_manifest_path = write_resolved_manifest(output_base, manifest, rows)
    metadata_path = write_run_metadata(
        output_base=output_base,
        manifest=manifest,
        rows=rows,
        galaxies=galaxies,
        selected_galaxy_names=args.galaxies,
        selected_config_names=args.config_name,
        warnings=warnings,
    )

    print(f"Original manifest copied to: {original_manifest_path}")
    print(f"Resolved manifest written to: {resolved_manifest_path}")
    print(f"Run metadata written to: {metadata_path}")

    start_time = time.time()
    for index, galaxy in enumerate(galaxies, start=1):
        print(f"\n[{index}/{len(galaxies)}] ", end="")
        try:
            process_galaxy(galaxy, rows, output_base)
        except Exception as error:
            print(f"  ERROR processing {galaxy.name}: {error}")
            continue

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Complete! Processed {len(galaxies)} galaxies in {elapsed:.1f} seconds")
    print(f"Output saved to: {output_base}")
    print(f"Total images generated: {len(galaxies) * len(rows)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
