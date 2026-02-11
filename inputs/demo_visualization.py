#!/usr/bin/env python
"""
Demonstration of the visualization functionality.

This script generates a mock galaxy image and creates a visualization with
arcsinh scaling and contours.
"""

from pathlib import Path
from mockgal import (
    MockGalaxy,
    SersicComponent,
    ImageConfig,
    MockImageGenerator,
    save_fits,
    visualize_galaxy
)


def main():
    """Generate and visualize a mock galaxy."""

    # Create a two-component galaxy (bulge + disk)
    print("Creating mock galaxy with bulge and disk components...")
    components = [
        # Bulge: compact, high Sersic index
        SersicComponent(
            r_eff_kpc=1.0,
            abs_mag=-19.5,
            n=4.0,
            ellipticity=0.2,
            pa_deg=45.0
        ),
        # Disk: extended, low Sersic index
        SersicComponent(
            r_eff_kpc=3.5,
            abs_mag=-20.8,
            n=1.0,
            ellipticity=0.6,
            pa_deg=45.0
        )
    ]

    galaxy = MockGalaxy(
        name="Demo Galaxy",
        redshift=0.01,
        components=components
    )

    # Configure image generation with PSF
    print("Configuring image generation...")
    config = ImageConfig(
        name="demo_clean",
        psf_enabled=True,
        psf_fwhm=1.0,
        size_pixels=501,
        engine="auto"
    )

    # Generate the image
    print("Generating mock galaxy image...")
    gen = MockImageGenerator(config)
    image, metadata = gen.generate(galaxy)

    # Save to FITS
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = output_dir / "demo_galaxy_demo_clean.fits"

    print(f"Saving FITS file to {fits_path}...")
    save_fits(image, metadata, fits_path)

    print(f"Image shape: {image.shape}")
    print(f"Max pixel value: {image.max():.2e}")
    print(f"Total flux: {image.sum():.2e}")

    # Create visualizations with different colormaps
    print("\nCreating visualizations...")

    colormaps = ['viridis', 'magma', 'inferno', 'cividis']

    for cmap in colormaps:
        png_path = fits_path.with_name(f"{fits_path.stem}_{cmap}.png")
        print(f"  Generating {cmap} visualization...")
        visualize_galaxy(
            fits_path,
            output_path=png_path,
            cmap=cmap,
            sigma_smooth=2.0,
            n_contours=8,
            dpi=150
        )

    print("\nVisualization complete!")
    print(f"FITS file: {fits_path}")
    print(f"PNG files: {output_dir}/demo_galaxy_demo_clean_*.png")


if __name__ == "__main__":
    main()
