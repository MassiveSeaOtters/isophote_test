#!/usr/bin/env python
"""Generate Huang2013 mock images with sky-sb-limit noise and PNG visualizations."""
from __future__ import annotations

import math
from pathlib import Path

from mockgal import (
    ImageConfig,
    MockGalaxy,
    MockImageGenerator,
    kpc_to_arcsec,
    load_model_file,
    save_fits,
    sb_limit_to_sigma,
    visualize_galaxy,
)

MODEL_PATH = Path("examples/huang2013_models.yaml")
OUTPUT_DIR = Path("output/huang2013_sblimit_noise_test")
TARGET_NAME = "NGC 3923"

REDSHIFT = 0.3
PIXEL_SCALE = 0.18  # arcsec/pixel
PSF_FWHM = 0.7      # arcsec
ZEROPOINT = 27.0
MAX_SIZE_PIXELS = 2000

SB_LIMITS = [23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
NOISE_SEED = 42


def _pa_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return 180.0 - d if d > 90.0 else d


def _size_from_max_re(max_re_kpc: float, redshift: float) -> int:
    re_arcsec = kpc_to_arcsec(max_re_kpc, redshift)
    re_pix = re_arcsec / PIXEL_SCALE
    target = math.ceil(10.0 * re_pix) + 1  # ensure strictly larger than 10x Re
    if target % 2 == 0:
        target += 1
    size = min(MAX_SIZE_PIXELS, target)
    if size % 2 == 0:
        size = size - 1 if size == MAX_SIZE_PIXELS else size + 1
    return size


def main() -> None:
    galaxies = load_model_file(str(MODEL_PATH), galaxy_names=[TARGET_NAME])
    if not galaxies:
        raise SystemExit(f"Galaxy not found: {TARGET_NAME}")

    base = galaxies[0]
    comps_sorted = sorted(base.components, key=lambda c: c.r_eff_kpc)
    central = comps_sorted[0]
    outer = comps_sorted[-1]

    ediff = abs(outer.ellipticity - central.ellipticity)
    pdiff = _pa_diff_deg(outer.pa_deg, central.pa_deg)
    if not (ediff > 0.15 and pdiff > 20.0):
        raise SystemExit(
            f"{TARGET_NAME} does not meet criteria: "
            f"ellip diff={ediff:.2f}, PA diff={pdiff:.2f} deg"
        )

    galaxy = MockGalaxy(name=base.name, redshift=REDSHIFT, components=base.components)
    max_re_kpc = max(c.r_eff_kpc for c in galaxy.components)
    size_pixels = _size_from_max_re(max_re_kpc, galaxy.redshift)

    out_dir = OUTPUT_DIR / TARGET_NAME.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Galaxy: {TARGET_NAME}")
    print(f"Central vs outer: ellip diff={ediff:.2f}, PA diff={pdiff:.2f} deg")
    print(f"Max Re (kpc): {max_re_kpc:.2f}")
    print(f"Image size: {size_pixels} x {size_pixels} pixels")

    configs = [("noiseless", None)] + [(f"sky_sblimit_{v:.1f}", v) for v in SB_LIMITS]
    for name, sb_limit in configs:
        cfg = ImageConfig(
            name=name,
            pixel_scale=PIXEL_SCALE,
            zeropoint=ZEROPOINT,
            size_pixels=size_pixels,
            psf_enabled=True,
            psf_type="gaussian",
            psf_fwhm=PSF_FWHM,
            noise_enabled=sb_limit is not None,
            sky_sb_limit=sb_limit,
            noise_seed=NOISE_SEED if sb_limit is not None else None,
            engine="auto",
        )

        generator = MockImageGenerator(config=cfg)
        image, metadata = generator.generate(galaxy)

        fits_path = out_dir / f"{TARGET_NAME.replace(' ', '_')}_{cfg.name}.fits"
        png_path = out_dir / f"{TARGET_NAME.replace(' ', '_')}_{cfg.name}.png"

        save_fits(image, metadata, fits_path)
        visualize_galaxy(
            fits_path,
            output_path=png_path,
            cmap="magma",
            sigma_smooth=2.0,
            n_contours=10,
            dpi=200,
        )
        fits_path.unlink(missing_ok=True)

        if sb_limit is None:
            print("  noiseless -> sigma=0")
        else:
            sigma = sb_limit_to_sigma(sb_limit, PIXEL_SCALE, ZEROPOINT)
            print(f"  sky-sb-limit={sb_limit:.1f} -> sigma={sigma:.3e} per pixel")

    print(f"PNGs saved to: {out_dir}")


if __name__ == "__main__":
    main()
