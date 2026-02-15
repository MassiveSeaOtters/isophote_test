#!/usr/bin/env python
"""
Generate systematic mock images for the entire Huang2013 sample.

Creates 4 versions of each galaxy with different redshifts and noise levels:
1. z=0.05, HSC pixel scale, FWHM=0.7", no noise
2. z=0.05, HSC pixel scale, FWHM=0.7", sky_sb_limit=24.5
3. z=0.20, HSC pixel scale, FWHM=0.7", sky_sb_limit=24.5
4. z=0.50, HSC pixel scale, FWHM=0.7", sky_sb_limit=24.5

Output structure:
    output_dir/
        huang2013/
            [GALAXY_NAME]/
                [GALAXY_NAME]_mock1.fits
                [GALAXY_NAME]_mock2.fits
                [GALAXY_NAME]_mock3.fits
                [GALAXY_NAME]_mock4.fits

Usage:
    python generate_huang2013_mocks.py --output /path/to/output
    python generate_huang2013_mocks.py --output /path/to/output --galaxies "NGC 3923" "IC 1459"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mockgal import (
    ImageConfig,
    MockGalaxy,
    MockImageGenerator,
    load_model_file,
    sanitize_filename,
    save_fits,
)

# HSC (Hyper Suprime-Cam) specifications
HSC_PIXEL_SCALE = 0.168  # arcsec/pixel
PSF_FWHM = 0.7          # arcsec
ZEROPOINT = 27.0        # magnitude
SIZE_FACTOR = 16.0      # image half-size = factor * max(Re)
SKY_SB_LIMIT = 24.5     # mag/arcsec^2
NOISE_SEED = 42

# Huang2013 model file
MODEL_FILE = Path(__file__).parent / "huang2013_models.yaml"


def create_image_configs() -> List[ImageConfig]:
    """
    Create the 4 image configurations.

    Returns
    -------
    list of ImageConfig
        Four configurations with different redshifts and noise settings
    """
    configs = [
        # Mock 1: z=0.05, no noise
        ImageConfig(
            name="mock1",
            pixel_scale=HSC_PIXEL_SCALE,
            zeropoint=ZEROPOINT,
            size_factor=SIZE_FACTOR,
            psf_enabled=True,
            psf_type="gaussian",
            psf_fwhm=PSF_FWHM,
            sky_enabled=False,
            noise_enabled=False,
            engine="auto",
        ),
        # Mock 2: z=0.05, with noise
        ImageConfig(
            name="mock2",
            pixel_scale=HSC_PIXEL_SCALE,
            zeropoint=ZEROPOINT,
            size_factor=SIZE_FACTOR,
            psf_enabled=True,
            psf_type="gaussian",
            psf_fwhm=PSF_FWHM,
            sky_enabled=False,
            noise_enabled=True,
            sky_sb_limit=SKY_SB_LIMIT,
            noise_seed=NOISE_SEED,
            engine="auto",
        ),
        # Mock 3: z=0.20, with noise
        ImageConfig(
            name="mock3",
            pixel_scale=HSC_PIXEL_SCALE,
            zeropoint=ZEROPOINT,
            size_factor=SIZE_FACTOR,
            psf_enabled=True,
            psf_type="gaussian",
            psf_fwhm=PSF_FWHM,
            sky_enabled=False,
            noise_enabled=True,
            sky_sb_limit=SKY_SB_LIMIT,
            noise_seed=NOISE_SEED,
            engine="auto",
        ),
        # Mock 4: z=0.50, with noise
        ImageConfig(
            name="mock4",
            pixel_scale=HSC_PIXEL_SCALE,
            zeropoint=ZEROPOINT,
            size_factor=SIZE_FACTOR,
            psf_enabled=True,
            psf_type="gaussian",
            psf_fwhm=PSF_FWHM,
            sky_enabled=False,
            noise_enabled=True,
            sky_sb_limit=SKY_SB_LIMIT,
            noise_seed=NOISE_SEED,
            engine="auto",
        ),
    ]

    return configs


def create_mosaic(
    galaxy_name: str,
    galaxy_dir: Path,
    redshifts: List[float],
) -> Optional[Path]:
    """
    Create a 2x2 mosaic of the 4 mock images for QA review.

    Parameters
    ----------
    galaxy_name : str
        Sanitized galaxy name (no spaces)
    galaxy_dir : Path
        Directory containing the 4 FITS files
    redshifts : list of float
        Redshifts for each mock configuration

    Returns
    -------
    Path or None
        Path to the saved mosaic PNG, or None if matplotlib unavailable
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.io import fits
    except ImportError:
        print("  Warning: matplotlib or astropy not available, skipping mosaic")
        return None

    labels = [
        f"mock1: z={redshifts[0]:.2f}, no noise",
        f"mock2: z={redshifts[1]:.2f}, sb_lim={SKY_SB_LIMIT}",
        f"mock3: z={redshifts[2]:.2f}, sb_lim={SKY_SB_LIMIT}",
        f"mock4: z={redshifts[3]:.2f}, sb_lim={SKY_SB_LIMIT}",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(galaxy_name, fontsize=16, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        mock_name = f"mock{idx + 1}"
        fits_path = galaxy_dir / f"{galaxy_name}_{mock_name}.fits"

        if not fits_path.exists():
            ax.set_title(f"{labels[idx]} (missing)", fontsize=10)
            ax.axis("off")
            continue

        with fits.open(fits_path) as hdul:
            image = hdul[0].data

        # Arcsinh scaling (same approach as mockgal.visualize_galaxy)
        scale = np.nanpercentile(image, 99.5) / 10.0
        if scale <= 0:
            scale = 1.0
        scaled = np.arcsinh(image / scale)

        ax.imshow(scaled, origin="lower", cmap="magma", interpolation="nearest")
        ax.set_title(labels[idx], fontsize=10)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = galaxy_dir / f"{galaxy_name}_mosaic.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def process_galaxy(
    galaxy: MockGalaxy,
    configs: List[ImageConfig],
    output_base: Path,
    redshifts: List[float],
) -> None:
    """
    Process a single galaxy through all 4 configurations.

    Parameters
    ----------
    galaxy : MockGalaxy
        Galaxy to process
    configs : list of ImageConfig
        Four image configurations
    output_base : Path
        Base output directory
    redshifts : list of float
        Redshifts for each configuration [z1, z2, z3, z4]
    """
    # Create galaxy-specific output directory
    safe_galaxy_name = sanitize_filename(galaxy.name)
    galaxy_dir = output_base / "huang2013" / safe_galaxy_name
    galaxy_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {galaxy.name}...")

    for i, (config, redshift) in enumerate(zip(configs, redshifts), 1):
        # Update galaxy redshift for this mock
        galaxy_at_z = MockGalaxy(
            name=galaxy.name,
            redshift=redshift,
            components=galaxy.components
        )

        # Generate image
        generator = MockImageGenerator(config)
        image, metadata = generator.generate(galaxy_at_z)

        # Save FITS file
        output_path = galaxy_dir / f"{safe_galaxy_name}_{config.name}.fits"
        save_fits(image, metadata, output_path)

        print(f"  [{i}/4] {config.name}: z={redshift:.2f}, "
              f"size={image.shape}, saved to {output_path.name}")

    # Generate QA mosaic
    mosaic_path = create_mosaic(safe_galaxy_name, galaxy_dir, redshifts)
    if mosaic_path:
        print(f"  Mosaic saved to {mosaic_path.name}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate systematic Huang2013 mock images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for mock images"
    )
    parser.add_argument(
        "--galaxies",
        nargs="+",
        type=str,
        default=None,
        help="Process only specific galaxies (default: all 93 galaxies)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first 2 galaxies"
    )

    args = parser.parse_args()

    # Load galaxy models
    print(f"Loading galaxy models from {MODEL_FILE}")
    galaxies = load_model_file(MODEL_FILE)
    print(f"Loaded {len(galaxies)} galaxies")

    # Filter galaxies if requested
    if args.galaxies:
        galaxy_names = set(args.galaxies)
        galaxies = [g for g in galaxies if g.name in galaxy_names]
        print(f"Processing {len(galaxies)} selected galaxies: {', '.join(args.galaxies)}")
    elif args.test:
        galaxies = galaxies[:2]
        print(f"Test mode: Processing first 2 galaxies: {galaxies[0].name}, {galaxies[1].name}")

    if not galaxies:
        print("Error: No galaxies to process!")
        return 1

    # Create configurations
    configs = create_image_configs()
    redshifts = [0.05, 0.05, 0.20, 0.50]  # Redshifts for mocks 1-4

    # Setup output directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Output directory: {output_base}")
    print(f"Number of galaxies: {len(galaxies)}")
    print(f"HSC pixel scale: {HSC_PIXEL_SCALE} arcsec/pixel")
    print(f"PSF FWHM: {PSF_FWHM} arcsec")
    print(f"Size factor: {SIZE_FACTOR} (capped at 4001 px)")
    print(f"Zeropoint: {ZEROPOINT} mag")
    print(f"Sky SB limit: {SKY_SB_LIMIT} mag/arcsec^2")
    print("\nMock configurations:")
    print("  Mock 1: z=0.05, no noise")
    print(f"  Mock 2: z=0.05, sky_sb_limit={SKY_SB_LIMIT}")
    print(f"  Mock 3: z=0.20, sky_sb_limit={SKY_SB_LIMIT}")
    print(f"  Mock 4: z=0.50, sky_sb_limit={SKY_SB_LIMIT}")
    print("=" * 70)

    # Process all galaxies
    start_time = time.time()

    for idx, galaxy in enumerate(galaxies, 1):
        print(f"\n[{idx}/{len(galaxies)}] ", end="")
        try:
            process_galaxy(galaxy, configs, output_base, redshifts)
        except Exception as e:
            print(f"  ERROR processing {galaxy.name}: {e}")
            continue

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Complete! Processed {len(galaxies)} galaxies in {elapsed:.1f} seconds")
    print(f"Output saved to: {output_base / 'huang2013'}")
    print(f"Total images generated: {len(galaxies) * 4}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
