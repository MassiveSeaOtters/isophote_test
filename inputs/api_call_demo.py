#!/usr/bin/env python
"""
Example script demonstrating direct API usage for mock image generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mockgal import (
    ImageConfig,
    SersicComponent,
    generate_mock_image,
    save_fits,
)


def main() -> None:
    components = [
        SersicComponent(
            r_eff_kpc=1.0,
            abs_mag=-20.0,
            n=4.0,
            ellipticity=0.2,
            pa_deg=30.0,
        )
    ]

    config = ImageConfig(size_pixels=51, engine="auto")

    image, metadata = generate_mock_image(
        name="api_demo",
        redshift=0.01,
        components=components,
        config=config,
        return_metadata=True,
    )

    print("Shape:", image.shape)
    print("Engine:", metadata.get("engine"))
    print("Max:", float(image.max()))

    save_fits(image, metadata, "output/api_demo.fits")


if __name__ == "__main__":
    main()
