#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip


DEFAULT_WIDE_CATALOG = Path("inputs/data/cosmos_wide_sky.csv")
DEFAULT_DUD_CATALOG = Path("inputs/data/cosmos_dud_sky.csv")
DEFAULT_OUTPUT_DIR = Path("output/hsc_sky_calibration")
DEFAULT_SIDE1_DIR = Path("../hsc_sandbox/side1")
DEFAULT_FILTER_NAME = "i"
DEFAULT_CUTOUT_TYPE = "coadd/bg"
DEFAULT_HALF_SIZE = "2.5arcsec"
DEFAULT_SAMPLE_COUNT = 30
DEFAULT_RANDOM_SEED = 42
DEFAULT_PIXEL_SCALE = 0.168
DEFAULT_ZEROPOINT = 27.0
DEFAULT_SIGMA_CLIP_SIGMA = 3.0
DEFAULT_SIGMA_CLIP_ITERS = 5

BASE_LAYER_CONFIGS = {
    "wide": {
        "catalog": DEFAULT_WIDE_CATALOG,
        "rerun": "s23b_wide",
    },
    "dud": {
        "catalog": DEFAULT_DUD_CATALOG,
        "rerun": "s21a_dud2",
    },
}


@dataclass(frozen=True, slots=True)
class SkyCatalogRow:
    object_id: str
    ra: float
    dec: float
    values: dict[str, str]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    label: str
    catalog_path: Path
    rerun: str


@dataclass(frozen=True, slots=True)
class SelectedSkyRegion:
    selection_id: str
    object_id: str
    ra: float
    dec: float


@dataclass(frozen=True, slots=True)
class CutoutMeasurement:
    selection_id: str
    object_id: str
    rerun: str
    filter_name: str
    image_type: str
    output_path: str
    width_pixels: int
    height_pixels: int
    finite_pixel_count: int
    variance_pixel_count: int
    local_background_median: float
    centered_mean: float
    centered_median: float
    centered_rms: float
    centered_robust_sigma: float
    variance_median: float | None
    variance_sigma: float | None
    background_sb_value: float | None


@dataclass(frozen=True, slots=True)
class LayerCalibrationSummary:
    label: str
    rerun: str
    filter_name: str
    image_type: str
    requested_cutout_count: int
    successful_cutout_count: int
    total_pooled_pixels: int
    total_clipped_pixels: int
    pooled_centered_mean: float
    pooled_centered_median: float
    pooled_centered_rms: float
    pooled_centered_robust_sigma: float
    variance_median: float | None
    variance_sigma: float | None
    variance_sigma_ratio: float | None
    recommended_sky_sb_limit: float | None
    original_background_median: float
    original_background_mean: float
    original_background_std: float
    positive_background_count: int
    exploratory_sky_sb_value: float | None


def normalize_header(value: str) -> str:
    return value.lstrip("# ").strip().lower()


def read_sky_catalog(path: Path) -> list[SkyCatalogRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[SkyCatalogRow] = []
        for row in reader:
            normalized = {
                normalize_header(key): (value or "").strip() for key, value in row.items() if key
            }
            rows.append(
                SkyCatalogRow(
                    object_id=normalized["object_id"],
                    ra=float(normalized["ra"]),
                    dec=float(normalized["dec"]),
                    values=normalized,
                )
            )
    return rows


def select_random_regions(
    rows: list[SkyCatalogRow],
    sample_count: int,
    random_seed: int,
) -> list[SelectedSkyRegion]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(rows):
        raise ValueError(
            f"sample_count={sample_count} exceeds available catalog size of {len(rows)}"
        )
    rng = np.random.default_rng(random_seed)
    selected_indices = sorted(rng.choice(len(rows), size=sample_count, replace=False).tolist())
    return [
        SelectedSkyRegion(
            selection_id=f"sky_{index + 1:02d}",
            object_id=rows[row_index].object_id,
            ra=rows[row_index].ra,
            dec=rows[row_index].dec,
        )
        for index, row_index in enumerate(selected_indices)
    ]


def write_selection_catalog(path: Path, regions: list[SelectedSkyRegion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "name", "ra", "dec", "object_id"],
        )
        writer.writeheader()
        for region in regions:
            writer.writerow(
                {
                    "id": region.selection_id,
                    "name": region.object_id,
                    "ra": f"{region.ra:.16f}",
                    "dec": f"{region.dec:.16f}",
                    "object_id": region.object_id,
                }
            )


def flux_to_surface_brightness(
    flux_per_pixel: float,
    pixel_scale: float,
    zeropoint: float,
) -> float | None:
    if flux_per_pixel <= 0:
        return None
    flux_per_arcsec2 = flux_per_pixel / (pixel_scale**2)
    return zeropoint - 2.5 * math.log10(flux_per_arcsec2)


def sigma_to_sb_limit(
    sigma: float,
    pixel_scale: float,
    zeropoint: float,
) -> float | None:
    if sigma <= 0:
        return None
    flux_5sigma = sigma * 5.0
    flux_per_arcsec2 = flux_5sigma / (pixel_scale**2)
    return zeropoint - 2.5 * math.log10(flux_per_arcsec2)


def robust_sigma(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def clip_values(values: np.ndarray, sigma_value: float, maxiters: int) -> np.ndarray:
    clipped = sigma_clip(values, sigma=sigma_value, maxiters=maxiters, masked=True)
    return np.ma.compressed(clipped)


def build_side1_cutout_argv(
    catalog_path: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    rerun: str,
    filter_name: str,
    image_type: str,
    half_size_width: str,
    half_size_height: str,
    overwrite: bool,
) -> list[str]:
    argv = [
        "cutout",
        "--input",
        str(catalog_path),
        "--input-format",
        "csv",
        "--filter",
        filter_name,
        "--rerun",
        rerun,
        "--half-size-width",
        half_size_width,
        "--half-size-height",
        half_size_height,
        "--type",
        image_type,
        "--image",
        "true",
        "--mask",
        "false",
        "--variance",
        "true",
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest_path),
    ]
    if overwrite:
        argv.append("--overwrite")
    return argv


def load_side1_run_command(side1_dir: Path):
    side1_python_dir = side1_dir / "python"
    if not side1_python_dir.is_dir():
        raise FileNotFoundError(f"side1 python directory not found: {side1_python_dir}")
    if str(side1_python_dir) not in sys.path:
        sys.path.insert(0, str(side1_python_dir))
    from hsc_image_psf_downloader import run_command  # noqa: PLC0415

    return run_command


def run_download_batch(
    side1_dir: Path,
    catalog_path: Path,
    output_dir: Path,
    *,
    rerun: str,
    filter_name: str,
    image_type: str,
    half_size_width: str,
    half_size_height: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.csv"
    run_command = load_side1_run_command(side1_dir)
    results = run_command(
        build_side1_cutout_argv(
            catalog_path,
            output_dir,
            manifest_path,
            rerun=rerun,
            filter_name=filter_name,
            image_type=image_type,
            half_size_width=half_size_width,
            half_size_height=half_size_height,
            overwrite=overwrite,
        )
    )
    return [result.to_manifest_row() for result in results]


def measure_cutout(
    fits_path: Path,
    *,
    sigma_clip_sigma: float,
    sigma_clip_iters: int,
    pixel_scale: float,
    zeropoint: float,
) -> tuple[CutoutMeasurement, np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(fits_path) as hdul:
        image = hdul[1].data.astype(np.float64, copy=False)
        variance_hdu = hdul[3].data
        if variance_hdu is None:
            raise ValueError(f"variance plane is missing in {fits_path}")
        variance = variance_hdu.astype(np.float64, copy=False)

    image_pixels = image[np.isfinite(image)]
    variance_pixels = variance[np.isfinite(variance) & (variance >= 0)]
    if image_pixels.size == 0:
        raise ValueError(f"image plane has no finite pixels in {fits_path}")
    if variance_pixels.size == 0:
        raise ValueError(f"variance plane has no finite non-negative pixels in {fits_path}")

    clipped_image_pixels = clip_values(image_pixels, sigma_clip_sigma, sigma_clip_iters)
    local_background_median = float(np.median(clipped_image_pixels))
    centered_image = image - local_background_median
    centered_pixels = centered_image[np.isfinite(centered_image)]
    clipped_centered_pixels = clip_values(centered_pixels, sigma_clip_sigma, sigma_clip_iters)

    measurement = CutoutMeasurement(
        selection_id="",
        object_id="",
        rerun="",
        filter_name="",
        image_type="",
        output_path=str(fits_path),
        width_pixels=int(image.shape[1]),
        height_pixels=int(image.shape[0]),
        finite_pixel_count=int(centered_pixels.size),
        variance_pixel_count=int(variance_pixels.size),
        local_background_median=local_background_median,
        centered_mean=float(np.mean(clipped_centered_pixels)),
        centered_median=float(np.median(clipped_centered_pixels)),
        centered_rms=float(np.std(clipped_centered_pixels)),
        centered_robust_sigma=float(robust_sigma(clipped_centered_pixels)),
        variance_median=float(np.median(variance_pixels)),
        variance_sigma=float(math.sqrt(max(float(np.median(variance_pixels)), 0.0))),
        background_sb_value=flux_to_surface_brightness(
            local_background_median,
            pixel_scale,
            zeropoint,
        ),
    )
    return measurement, centered_image, clipped_centered_pixels, variance_pixels


def finalize_measurement(
    measurement: CutoutMeasurement,
    result_row: dict[str, Any],
    selection_lookup: dict[str, SelectedSkyRegion],
) -> CutoutMeasurement:
    selection_id = result_row["object_id"]
    selection = selection_lookup[selection_id]
    return CutoutMeasurement(
        selection_id=selection_id,
        object_id=selection.object_id,
        rerun=result_row["rerun"],
        filter_name=result_row["filter_name"],
        image_type=result_row["image_type"],
        output_path=result_row["output_path"],
        width_pixels=measurement.width_pixels,
        height_pixels=measurement.height_pixels,
        finite_pixel_count=measurement.finite_pixel_count,
        variance_pixel_count=measurement.variance_pixel_count,
        local_background_median=measurement.local_background_median,
        centered_mean=measurement.centered_mean,
        centered_median=measurement.centered_median,
        centered_rms=measurement.centered_rms,
        centered_robust_sigma=measurement.centered_robust_sigma,
        variance_median=measurement.variance_median,
        variance_sigma=measurement.variance_sigma,
        background_sb_value=measurement.background_sb_value,
    )


def summarize_layer(
    label: str,
    rerun: str,
    filter_name: str,
    image_type: str,
    requested_cutout_count: int,
    measurements: list[CutoutMeasurement],
    pooled_pixels: np.ndarray,
    pooled_variance_pixels: np.ndarray,
    *,
    sigma_clip_sigma: float,
    sigma_clip_iters: int,
    pixel_scale: float,
    zeropoint: float,
) -> LayerCalibrationSummary:
    clipped_pooled_pixels = clip_values(pooled_pixels, sigma_clip_sigma, sigma_clip_iters)
    total_clipped_pixels = int(pooled_pixels.size - clipped_pooled_pixels.size)
    variance_median = (
        float(np.median(pooled_variance_pixels)) if pooled_variance_pixels.size > 0 else None
    )
    variance_sigma = (
        float(math.sqrt(max(variance_median, 0.0))) if variance_median is not None else None
    )

    local_background_medians = np.array(
        [measurement.local_background_median for measurement in measurements],
        dtype=np.float64,
    )
    positive_backgrounds = local_background_medians[local_background_medians > 0]
    pooled_rms = float(np.std(clipped_pooled_pixels))

    variance_sigma_ratio = None
    if variance_sigma is not None and variance_sigma > 0:
        variance_sigma_ratio = pooled_rms / variance_sigma

    exploratory_sky_sb_value = None
    if positive_backgrounds.size > 0:
        exploratory_sky_sb_value = flux_to_surface_brightness(
            float(np.median(positive_backgrounds)),
            pixel_scale,
            zeropoint,
        )

    return LayerCalibrationSummary(
        label=label,
        rerun=rerun,
        filter_name=filter_name,
        image_type=image_type,
        requested_cutout_count=requested_cutout_count,
        successful_cutout_count=len(measurements),
        total_pooled_pixels=int(pooled_pixels.size),
        total_clipped_pixels=total_clipped_pixels,
        pooled_centered_mean=float(np.mean(clipped_pooled_pixels)),
        pooled_centered_median=float(np.median(clipped_pooled_pixels)),
        pooled_centered_rms=pooled_rms,
        pooled_centered_robust_sigma=float(robust_sigma(clipped_pooled_pixels)),
        variance_median=variance_median,
        variance_sigma=variance_sigma,
        variance_sigma_ratio=variance_sigma_ratio,
        recommended_sky_sb_limit=sigma_to_sb_limit(pooled_rms, pixel_scale, zeropoint),
        original_background_median=float(np.median(local_background_medians)),
        original_background_mean=float(np.mean(local_background_medians)),
        original_background_std=float(np.std(local_background_medians)),
        positive_background_count=int(positive_backgrounds.size),
        exploratory_sky_sb_value=exploratory_sky_sb_value,
    )


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV without fieldnames: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def import_matplotlib():
    import matplotlib.pyplot as plt  # noqa: PLC0415

    return plt


def choose_display_limits(
    real_images: list[np.ndarray],
    synthetic_images: list[np.ndarray],
) -> tuple[float, float]:
    combined = np.concatenate(
        [np.ravel(image) for image in [*real_images, *synthetic_images] if image.size > 0]
    )
    scale = float(np.percentile(np.abs(combined), 99))
    return (-scale, scale)


def plot_image_grid(
    images: list[np.ndarray],
    output_path: Path,
    *,
    title: str,
    vmin: float,
    vmax: float,
    panel_titles: list[str] | None = None,
) -> None:
    plt = import_matplotlib()
    image_count = len(images)
    cols = min(6, image_count)
    rows = math.ceil(image_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= image_count:
            continue
        axis.imshow(images[index], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        if panel_titles:
            axis.set_title(panel_titles[index], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_side_by_side_grid(
    real_images: list[np.ndarray],
    synthetic_images: list[np.ndarray],
    output_path: Path,
    *,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    paired_images = [np.hstack([real_image, synthetic_image]) for real_image, synthetic_image in zip(real_images, synthetic_images, strict=True)]
    plot_image_grid(
        paired_images,
        output_path,
        title=title,
        vmin=vmin,
        vmax=vmax,
    )


def plot_pixel_distribution(
    real_pixels: np.ndarray,
    synthetic_pixels: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    plt = import_matplotlib()
    fig, axis = plt.subplots(figsize=(8, 5))
    bins = 80
    axis.hist(real_pixels, bins=bins, density=True, histtype="step", linewidth=1.8, label="Real sky")
    axis.hist(
        synthetic_pixels,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Synthetic noise",
    )
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Centered pixel value")
    axis.set_ylabel("Density")
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_synthetic_noise_images(
    shapes: list[tuple[int, int]],
    sigma_value: float,
    random_seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(random_seed)
    images = [rng.normal(0.0, sigma_value, size=shape).astype(np.float64) for shape in shapes]
    pooled_pixels = np.concatenate([image.ravel() for image in images])
    return images, pooled_pixels


def run_dataset(
    dataset: DatasetSpec,
    args: argparse.Namespace,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    catalog_rows = read_sky_catalog(dataset.catalog_path)
    selected_regions = select_random_regions(
        catalog_rows,
        args.sample_count,
        args.random_seed + seed_offset,
    )
    selection_lookup = {region.selection_id: region for region in selected_regions}

    dataset_dir = args.output_dir / dataset.label
    selection_catalog_path = dataset_dir / f"selected_regions_{dataset.label}.csv"
    write_selection_catalog(selection_catalog_path, selected_regions)

    manifest_rows = run_download_batch(
        args.side1_dir.resolve(),
        selection_catalog_path,
        dataset_dir / "downloads",
        rerun=dataset.rerun,
        filter_name=args.filter_name,
        image_type=args.cutout_type,
        half_size_width=args.half_size_width,
        half_size_height=args.half_size_height,
        overwrite=args.overwrite,
    )
    write_csv_rows(dataset_dir / f"download_manifest_{dataset.label}.csv", manifest_rows)

    success_rows = [row for row in manifest_rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError(f"no cutouts downloaded successfully for {dataset.label}")
    if args.download_only:
        return {
            "selection_catalog": str(selection_catalog_path),
            "manifest_rows": manifest_rows,
            "measurements": [],
            "summary": {
                "label": dataset.label,
                "rerun": dataset.rerun,
                "filter_name": args.filter_name,
                "image_type": args.cutout_type,
                "requested_cutout_count": args.sample_count,
                "successful_cutout_count": len(success_rows),
            },
        }

    measurements: list[CutoutMeasurement] = []
    centered_images: list[np.ndarray] = []
    pooled_pixel_sets: list[np.ndarray] = []
    pooled_variance_sets: list[np.ndarray] = []
    for row in success_rows:
        measurement, centered_image, clipped_pixels, variance_pixels = measure_cutout(
            Path(row["output_path"]),
            sigma_clip_sigma=args.sigma_clip_sigma,
            sigma_clip_iters=args.sigma_clip_iters,
            pixel_scale=args.pixel_scale,
            zeropoint=args.zeropoint,
        )
        measurements.append(finalize_measurement(measurement, row, selection_lookup))
        centered_images.append(centered_image)
        pooled_pixel_sets.append(clipped_pixels)
        pooled_variance_sets.append(variance_pixels)

    measurement_rows = [asdict(measurement) for measurement in measurements]
    write_csv_rows(dataset_dir / f"cutout_stats_{dataset.label}.csv", measurement_rows)

    pooled_pixels = np.concatenate(pooled_pixel_sets)
    pooled_variance_pixels = np.concatenate(pooled_variance_sets)
    summary = summarize_layer(
        dataset.label,
        dataset.rerun,
        args.filter_name,
        args.cutout_type,
        args.sample_count,
        measurements,
        pooled_pixels,
        pooled_variance_pixels,
        sigma_clip_sigma=args.sigma_clip_sigma,
        sigma_clip_iters=args.sigma_clip_iters,
        pixel_scale=args.pixel_scale,
        zeropoint=args.zeropoint,
    )

    summary_dict = asdict(summary)
    write_json(dataset_dir / f"layer_summary_{dataset.label}.json", summary_dict)
    write_csv_rows(dataset_dir / f"layer_summary_{dataset.label}.csv", [summary_dict])

    if args.make_qa:
        shapes = [(image.shape[0], image.shape[1]) for image in centered_images]
        synthetic_images, synthetic_pixels = make_synthetic_noise_images(
            shapes,
            summary.pooled_centered_rms,
            args.random_seed + 1000 + seed_offset,
        )
        vmin, vmax = choose_display_limits(centered_images, synthetic_images)
        panel_titles = [measurement.selection_id for measurement in measurements]

        plot_image_grid(
            centered_images,
            dataset_dir / f"qa_real_cutouts_{dataset.label}.png",
            title=f"{dataset.label} centered HSC sky cutouts",
            vmin=vmin,
            vmax=vmax,
            panel_titles=panel_titles,
        )
        plot_image_grid(
            synthetic_images,
            dataset_dir / f"qa_simulated_noise_{dataset.label}.png",
            title=f"{dataset.label} synthetic MockGal-style noise",
            vmin=vmin,
            vmax=vmax,
            panel_titles=panel_titles,
        )
        plot_side_by_side_grid(
            centered_images,
            synthetic_images,
            dataset_dir / f"qa_side_by_side_{dataset.label}.png",
            title=f"{dataset.label} real vs synthetic background",
            vmin=vmin,
            vmax=vmax,
        )
        plot_pixel_distribution(
            pooled_pixels,
            synthetic_pixels,
            dataset_dir / f"qa_pixel_distribution_{dataset.label}.png",
            title=f"{dataset.label} centered pixel distribution",
        )

    return {
        "selection_catalog": str(selection_catalog_path),
        "manifest_rows": manifest_rows,
        "measurements": measurement_rows,
        "summary": summary_dict,
    }


def resolve_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    layer_configs = {
        "wide": {"catalog": args.wide_catalog, "rerun": BASE_LAYER_CONFIGS["wide"]["rerun"]},
        "dud": {"catalog": args.dud_catalog, "rerun": BASE_LAYER_CONFIGS["dud"]["rerun"]},
    }
    if args.catalog is not None:
        if not args.rerun:
            raise ValueError("--rerun is required when --catalog is provided")
        label = args.label or args.catalog.stem
        return [DatasetSpec(label=label, catalog_path=args.catalog, rerun=args.rerun)]

    if args.layer == "both":
        return [
            DatasetSpec(
                label=label,
                catalog_path=config["catalog"],
                rerun=config["rerun"],
            )
            for label, config in layer_configs.items()
        ]

    config = layer_configs[args.layer]
    return [DatasetSpec(label=args.layer, catalog_path=config["catalog"], rerun=config["rerun"])]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate HSC sky background noise from empty-region catalogs."
    )
    parser.add_argument("--layer", choices=("wide", "dud", "both"), default="both")
    parser.add_argument("--catalog", type=Path, help="Custom sky-region catalog for a single run.")
    parser.add_argument("--rerun", help="HSC rerun for a custom catalog.")
    parser.add_argument("--label", help="Output label for a custom catalog run.")
    parser.add_argument("--wide-catalog", type=Path, default=DEFAULT_WIDE_CATALOG)
    parser.add_argument("--dud-catalog", type=Path, default=DEFAULT_DUD_CATALOG)
    parser.add_argument("--side1-dir", type=Path, default=DEFAULT_SIDE1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--filter", dest="filter_name", default=DEFAULT_FILTER_NAME)
    parser.add_argument("--cutout-type", default=DEFAULT_CUTOUT_TYPE)
    parser.add_argument("--half-size-width", default=DEFAULT_HALF_SIZE)
    parser.add_argument("--half-size-height", default=DEFAULT_HALF_SIZE)
    parser.add_argument("--sigma-clip-sigma", type=float, default=DEFAULT_SIGMA_CLIP_SIGMA)
    parser.add_argument("--sigma-clip-iters", type=int, default=DEFAULT_SIGMA_CLIP_ITERS)
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--zeropoint", type=float, default=DEFAULT_ZEROPOINT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-qa", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.make_qa = not args.no_qa

    datasets = resolve_datasets(args)
    combined_summary_rows: list[dict[str, Any]] = []
    for seed_offset, dataset in enumerate(datasets):
        result = run_dataset(dataset, args, seed_offset=seed_offset)
        combined_summary_rows.append(result["summary"])
        summary = result["summary"]
        print(
            f"{dataset.label}: successes={summary['successful_cutout_count']}/"
            f"{summary['requested_cutout_count']} pooled_rms={summary['pooled_centered_rms']:.6g} "
            f"sky_sb_limit={summary['recommended_sky_sb_limit']}"
        )

    if len(combined_summary_rows) > 1:
        write_csv_rows(args.output_dir / "combined_summary.csv", combined_summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
