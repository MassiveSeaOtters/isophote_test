from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hsc_sky_calibration import (  # noqa: E402
    BASE_LAYER_CONFIGS,
    CutoutMeasurement,
    build_side1_cutout_argv,
    flux_to_surface_brightness,
    measure_cutout,
    read_sky_catalog,
    resolve_datasets,
    select_random_regions,
    sigma_to_sb_limit,
    summarize_layer,
)


def test_read_sky_catalog_normalizes_header() -> None:
    rows = read_sky_catalog(Path("inputs/data/cosmos_wide_sky.csv"))

    assert rows
    assert rows[0].object_id
    assert isinstance(rows[0].ra, float)
    assert isinstance(rows[0].dec, float)


def test_select_random_regions_is_reproducible() -> None:
    rows = read_sky_catalog(Path("inputs/data/cosmos_wide_sky.csv"))[:50]

    first = select_random_regions(rows, sample_count=5, random_seed=11)
    second = select_random_regions(rows, sample_count=5, random_seed=11)

    assert [row.object_id for row in first] == [row.object_id for row in second]
    assert len(first) == 5


def test_build_side1_cutout_argv_uses_image_and_variance_only() -> None:
    argv = build_side1_cutout_argv(
        Path("selected.csv"),
        Path("output"),
        Path("output/download_manifest.csv"),
        rerun="s21a_dud2",
        filter_name="i",
        image_type="coadd/bg",
        half_size_width="2.5arcsec",
        half_size_height="2.5arcsec",
        overwrite=True,
    )

    assert "--image" in argv and "true" in argv
    assert "--mask" in argv and "false" in argv
    assert "--variance" in argv and "true" in argv
    assert "--type" in argv and "coadd/bg" in argv
    assert "--overwrite" in argv


def test_resolve_datasets_uses_canonical_layer_defaults() -> None:
    args = Namespace(
        catalog=None,
        rerun=None,
        label=None,
        layer="both",
        wide_catalog=Path("inputs/data/cosmos_wide_sky.csv"),
        dud_catalog=Path("inputs/data/cosmos_dud_sky.csv"),
    )

    datasets = resolve_datasets(args)

    assert [(dataset.label, dataset.rerun) for dataset in datasets] == [
        ("wide", BASE_LAYER_CONFIGS["wide"]["rerun"]),
        ("dud", BASE_LAYER_CONFIGS["dud"]["rerun"]),
    ]


def test_measure_cutout_recenters_background_and_reads_variance(tmp_path: Path) -> None:
    image = np.array(
        [
            [10.0, 10.2, 9.8, 10.1],
            [9.9, 10.0, 10.1, 10.2],
            [10.0, 9.8, 10.1, 10.0],
            [10.2, 9.9, 10.0, 10.1],
        ]
    )
    variance = np.full_like(image, 0.04)
    fits_path = tmp_path / "cutout.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image),
            fits.ImageHDU(),
            fits.ImageHDU(variance),
        ]
    ).writeto(fits_path)

    measurement, centered_image, clipped_pixels, variance_pixels = measure_cutout(
        fits_path,
        sigma_clip_sigma=3.0,
        sigma_clip_iters=5,
        pixel_scale=0.168,
        zeropoint=27.0,
    )

    assert measurement.local_background_median == pytest.approx(10.0)
    assert np.median(centered_image) == pytest.approx(0.0)
    assert measurement.variance_sigma == pytest.approx(0.2)
    assert clipped_pixels.size == image.size
    assert variance_pixels.size == variance.size


def test_summarize_layer_returns_mockgal_calibration_values() -> None:
    measurements = [
        CutoutMeasurement(
            selection_id="sky_01",
            object_id="obj_1",
            rerun="s23b_wide",
            filter_name="i",
            image_type="coadd/bg",
            output_path="one.fits",
            width_pixels=4,
            height_pixels=4,
            finite_pixel_count=16,
            variance_pixel_count=16,
            local_background_median=0.02,
            centered_mean=0.0,
            centered_median=0.0,
            centered_rms=0.05,
            centered_robust_sigma=0.05,
            variance_median=0.0025,
            variance_sigma=0.05,
            background_sb_value=flux_to_surface_brightness(0.02, 0.168, 27.0),
        ),
        CutoutMeasurement(
            selection_id="sky_02",
            object_id="obj_2",
            rerun="s23b_wide",
            filter_name="i",
            image_type="coadd/bg",
            output_path="two.fits",
            width_pixels=4,
            height_pixels=4,
            finite_pixel_count=16,
            variance_pixel_count=16,
            local_background_median=-0.01,
            centered_mean=0.0,
            centered_median=0.0,
            centered_rms=0.05,
            centered_robust_sigma=0.05,
            variance_median=0.0025,
            variance_sigma=0.05,
            background_sb_value=None,
        ),
    ]
    pooled_pixels = np.array([-0.05, -0.02, 0.0, 0.03, 0.05, 0.02, -0.01, 0.01])
    pooled_variance = np.full(8, 0.0025)

    summary = summarize_layer(
        "wide",
        "s23b_wide",
        "i",
        "coadd/bg",
        requested_cutout_count=2,
        measurements=measurements,
        pooled_pixels=pooled_pixels,
        pooled_variance_pixels=pooled_variance,
        sigma_clip_sigma=3.0,
        sigma_clip_iters=5,
        pixel_scale=0.168,
        zeropoint=27.0,
    )

    assert summary.successful_cutout_count == 2
    assert summary.recommended_sky_sb_limit == pytest.approx(
        sigma_to_sb_limit(summary.pooled_centered_rms, 0.168, 27.0)
    )
    assert summary.positive_background_count == 1
    assert summary.exploratory_sky_sb_value is not None
