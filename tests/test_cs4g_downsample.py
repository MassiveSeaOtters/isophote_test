from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "inputs"
    / "cs4g"
    / "scripts"
    / "downsample.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "cs4g_downsample_module",
    MODULE_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
downsample = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = downsample
MODULE_SPEC.loader.exec_module(downsample)


def test_rendered_size_pixels_matches_minimum_floor() -> None:
    size = downsample.rendered_size_pixels(
        max_component_extent_kpc=0.01,
        redshift=0.1,
        pixel_scale=0.168,
        size_factor=4.0,
    )

    assert size == downsample.MIN_IMAGE_EXTENT_PIX


def test_rendered_size_pixels_grows_with_component_extent() -> None:
    small = downsample.rendered_size_pixels(
        max_component_extent_kpc=1.0,
        redshift=0.1,
        pixel_scale=0.168,
        size_factor=4.0,
    )
    large = downsample.rendered_size_pixels(
        max_component_extent_kpc=3.0,
        redshift=0.1,
        pixel_scale=0.168,
        size_factor=4.0,
    )

    assert large > small


def test_select_sample_prefers_complexity_then_size_then_mass() -> None:
    parent = [
        {
            "name": "largest_two_component",
            "complexity_rank": 3,
            "n_components": 2,
            "size_pixels_cut": 301,
            "logmstar": 10.0,
        },
        {
            "name": "smaller_three_component",
            "complexity_rank": 3,
            "n_components": 3,
            "size_pixels_cut": 101,
            "logmstar": 9.0,
        },
        {
            "name": "larger_three_component",
            "complexity_rank": 3,
            "n_components": 3,
            "size_pixels_cut": 151,
            "logmstar": 8.5,
        },
        {
            "name": "lower_rank_four_component",
            "complexity_rank": 2,
            "n_components": 4,
            "size_pixels_cut": 999,
            "logmstar": 11.0,
        },
        {
            "name": "same_size_higher_mass",
            "complexity_rank": 3,
            "n_components": 3,
            "size_pixels_cut": 151,
            "logmstar": 10.5,
        },
    ]

    selected = downsample.select_sample(parent, target_n=5)

    assert [galaxy["name"] for galaxy in selected] == [
        "same_size_higher_mass",
        "larger_three_component",
        "smaller_three_component",
        "largest_two_component",
        "lower_rank_four_component",
    ]
