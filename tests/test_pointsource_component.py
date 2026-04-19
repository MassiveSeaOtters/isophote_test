"""Unit tests for PointSourceComponent (mockgal Phase D)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mockgal import (
    Component,
    ImageConfig,
    MockGalaxy,
    PointSourceComponent,
    RenderContext,
    SersicComponent,
    _build_component_from_dict,
    abs_to_app_mag,
    generate_mock_image,
    load_model_file,
)


def _ctx(redshift=0.01, pixel_scale=0.168, zeropoint=27.0,
         xcen=100.0, ycen=100.0):
    return RenderContext(
        redshift=redshift,
        pixel_scale_arcsec_per_pix=pixel_scale,
        zeropoint=zeropoint,
        xcen_pix=xcen,
        ycen_pix=ycen,
    )


class TestPointSourceValidation:
    def test_default_construction(self):
        ps = PointSourceComponent(abs_mag=-15.0)
        assert ps.abs_mag == -15.0
        assert ps.x_offset_pix is None
        assert ps.y_offset_pix is None
        assert ps.x_offset_kpc is None
        assert ps.y_offset_kpc is None

    def test_pix_offset_construction(self):
        ps = PointSourceComponent(abs_mag=-15.0, x_offset_pix=2.5, y_offset_pix=-1.0)
        assert ps.x_offset_pix == 2.5
        assert ps.y_offset_pix == -1.0

    def test_kpc_offset_construction(self):
        ps = PointSourceComponent(abs_mag=-15.0, x_offset_kpc=0.5, y_offset_kpc=-0.2)
        assert ps.x_offset_kpc == 0.5
        assert ps.y_offset_kpc == -0.2

    def test_x_pix_and_kpc_mutex(self):
        with pytest.raises(ValueError, match="Provide x_offset_pix OR"):
            PointSourceComponent(abs_mag=-15.0, x_offset_pix=1.0, x_offset_kpc=0.5)

    def test_y_pix_and_kpc_mutex(self):
        with pytest.raises(ValueError, match="Provide y_offset_pix OR"):
            PointSourceComponent(abs_mag=-15.0, y_offset_pix=1.0, y_offset_kpc=0.5)


class TestPointSourceComponentABC:
    def test_is_a_component(self):
        assert issubclass(PointSourceComponent, Component)

    def test_implements_all_abstract_methods(self):
        ps = PointSourceComponent(abs_mag=-15.0)
        ctx = _ctx()
        assert isinstance(ps.to_libprofit_spec(ctx), str)
        assert isinstance(ps.derived_params(ctx), dict)
        assert isinstance(ps.angular_extent_arcsec(ctx), float)
        assert isinstance(ps.to_astropy_image(ctx, (101, 101)), np.ndarray)


class TestPointSourceLibprofitSpec:
    def test_spec_at_center(self):
        ps = PointSourceComponent(abs_mag=-15.0)
        ctx = _ctx(xcen=100.0, ycen=100.0)
        spec = ps.to_libprofit_spec(ctx)
        assert spec.startswith("psf:")
        assert "xcen=100.0" in spec
        assert "ycen=100.0" in spec
        assert f"mag={abs_to_app_mag(-15.0, 0.01)}" in spec

    def test_spec_with_pix_offset(self):
        ps = PointSourceComponent(abs_mag=-15.0, x_offset_pix=2.0, y_offset_pix=-3.0)
        spec = ps.to_libprofit_spec(_ctx(xcen=100.0, ycen=100.0))
        assert "xcen=102.0" in spec
        assert "ycen=97.0" in spec

    def test_spec_with_kpc_offset_converts(self):
        ps = PointSourceComponent(abs_mag=-15.0, x_offset_kpc=0.0)
        spec = ps.to_libprofit_spec(_ctx(xcen=50.0))
        assert "xcen=50.0" in spec  # 0 kpc -> 0 pix offset


class TestPointSourceAstropyDelta:
    def test_extent_is_zero(self):
        ps = PointSourceComponent(abs_mag=-15.0)
        assert ps.angular_extent_arcsec(_ctx()) == 0.0

    def test_stamp_at_center(self):
        ps = PointSourceComponent(abs_mag=-15.0)
        ctx = _ctx(zeropoint=27.0, xcen=50.0, ycen=50.0)
        img = ps.to_astropy_image(ctx, (101, 101))
        # All flux at one pixel at (50, 50)
        nonzero = np.argwhere(img > 0)
        assert len(nonzero) == 1
        assert tuple(nonzero[0]) == (50, 50)
        # Total flux equals 10**(-0.4*(app_mag - zp))
        app_mag = abs_to_app_mag(-15.0, 0.01)
        expected = 10 ** (-0.4 * (app_mag - 27.0))
        assert img[50, 50] == pytest.approx(expected, rel=1e-12)

    def test_stamp_with_pix_offset(self):
        ps = PointSourceComponent(abs_mag=-15.0, x_offset_pix=3.0, y_offset_pix=-2.0)
        ctx = _ctx(xcen=50.0, ycen=50.0)
        img = ps.to_astropy_image(ctx, (101, 101))
        nonzero = np.argwhere(img > 0)
        assert len(nonzero) == 1
        # NumPy is (row, col) = (y, x); row 48 = ycen-2, col 53 = xcen+3
        assert tuple(nonzero[0]) == (48, 53)

    def test_offcenter_outside_image_bounds_silently_skips(self):
        ps = PointSourceComponent(abs_mag=-15.0, x_offset_pix=200.0)
        ctx = _ctx(xcen=50.0, ycen=50.0)
        img = ps.to_astropy_image(ctx, (51, 51))
        assert img.sum() == 0.0


class TestPointSourceGuard:
    def test_psf_disabled_with_psf_component_raises(self):
        config = ImageConfig(size_pixels=51, psf_enabled=False, engine="astropy")
        with pytest.raises(ValueError, match="PointSourceComponent requires"):
            generate_mock_image(
                name="psf_only",
                redshift=0.01,
                components=[PointSourceComponent(abs_mag=-15.0)],
                config=config,
            )

    def test_psf_enabled_psf_only_galaxy_renders(self):
        config = ImageConfig(
            size_pixels=51, engine="astropy",
            psf_enabled=True, psf_type="gaussian", psf_fwhm=1.0,
        )
        img, meta = generate_mock_image(
            name="psf_only",
            redshift=0.01,
            components=[PointSourceComponent(abs_mag=-15.0)],
            config=config,
        )
        assert img.shape == (51, 51)
        # Total flux conserved up to numerical precision (PSF normalized to 1)
        app_mag = abs_to_app_mag(-15.0, 0.01)
        expected_flux = 10 ** (-0.4 * (app_mag - 27.0))
        assert img.sum() == pytest.approx(expected_flux, rel=1e-3)

    def test_mixed_sersic_plus_psf_galaxy(self):
        config = ImageConfig(
            size_pixels=101, engine="astropy",
            psf_enabled=True, psf_type="gaussian", psf_fwhm=1.5,
        )
        img, meta = generate_mock_image(
            name="bulge_plus_nucleus",
            redshift=0.01,
            components=[
                SersicComponent(r_eff_kpc=2.0, abs_mag=-19.0, n=4.0),
                PointSourceComponent(abs_mag=-16.0),
            ],
            config=config,
        )
        assert img.shape == (101, 101)
        assert img.max() > 0
        assert np.isfinite(img).all()


class TestPointSourceImageSizeFallback:
    def test_psf_only_galaxy_auto_size(self):
        """No size_pixels — galaxy has no intrinsic extent, so the
        auto-size path must fall back to MIN_IMAGE_EXTENT_PIX (51)."""
        config = ImageConfig(
            engine="astropy",
            psf_enabled=True, psf_type="gaussian", psf_fwhm=1.0,
        )
        img, meta = generate_mock_image(
            name="psf_only",
            redshift=0.01,
            components=[PointSourceComponent(abs_mag=-15.0)],
            config=config,
        )
        # Shape must be at least the minimum, not collapsed to size=1.
        assert img.shape[0] >= 51
        assert img.shape[1] >= 51


class TestPointSourceRegistry:
    def test_built_from_dict_default(self):
        c = _build_component_from_dict({"type": "psf", "abs_mag": -15.0})
        assert isinstance(c, PointSourceComponent)
        assert c.abs_mag == -15.0

    def test_built_from_dict_with_offsets(self):
        c = _build_component_from_dict({
            "type": "psf",
            "abs_mag": -15.0,
            "x_offset_pix": 2.0,
            "y_offset_pix": -1.0,
        })
        assert c.x_offset_pix == 2.0
        assert c.y_offset_pix == -1.0


class TestPointSourceYamlRoundtrip:
    def test_load_from_yaml_manifest(self, tmp_path):
        manifest = tmp_path / "psf_manifest.yaml"
        manifest.write_text(yaml.safe_dump({
            "galaxies": [{
                "name": "agn_only",
                "redshift": 0.01,
                "components": [
                    {
                        "type": "psf",
                        "id": "nucleus",
                        "abs_mag": -16.5,
                    },
                ],
            }],
        }))
        galaxies = load_model_file(str(manifest))
        assert len(galaxies) == 1
        gal = galaxies[0]
        assert isinstance(gal, MockGalaxy)
        assert gal.name == "agn_only"
        assert len(gal.components) == 1
        assert isinstance(gal.components[0], PointSourceComponent)
        assert gal.components[0].component_id == "nucleus"
