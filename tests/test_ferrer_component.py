"""Unit tests for FerrerComponent (mockgal Phase C).

The Ferrer profile is a libprofit-only profile in mockgal — astropy has
no native Ferrer renderer, so to_astropy_image must raise
NotImplementedError. The integrated-magnitude semantics are checked by
asserting the libprofit profile-spec string contains the expected
``mag=`` value (full flux render against profit-cli is exercised
end-to-end in test_mockgal_galfit.py).
"""

import sys
from pathlib import Path

import pytest
import yaml

# Make the repo importable when pytest runs from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mockgal import (
    Component,
    FerrerComponent,
    ImageConfig,
    MockGalaxy,
    RenderContext,
    _build_component_from_dict,
    abs_to_app_mag,
    generate_mock_image,
    kpc_to_arcsec,
    load_model_file,
)


def _ctx(redshift=0.01, pixel_scale=0.168, zeropoint=27.0):
    return RenderContext(
        redshift=redshift,
        pixel_scale_arcsec_per_pix=pixel_scale,
        zeropoint=zeropoint,
        xcen_pix=100.0,
        ycen_pix=100.0,
    )


class TestFerrerValidation:
    def test_valid_construction(self):
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0)
        assert fc.r_out_kpc == 2.0
        assert fc.abs_mag == -18.0
        assert fc.alpha == 2.0
        assert fc.beta == 0.0
        assert fc.ellipticity == 0.0
        assert fc.pa_deg == 0.0

    def test_negative_r_out_rejected(self):
        with pytest.raises(ValueError, match="r_out_kpc must be positive"):
            FerrerComponent(r_out_kpc=-1.0, abs_mag=-18.0)

    def test_zero_r_out_rejected(self):
        with pytest.raises(ValueError, match="r_out_kpc must be positive"):
            FerrerComponent(r_out_kpc=0.0, abs_mag=-18.0)

    def test_negative_alpha_rejected(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0, alpha=-1.0)

    def test_zero_alpha_rejected(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0, alpha=0.0)

    def test_ellipticity_one_rejected(self):
        with pytest.raises(ValueError, match="Ellipticity must be"):
            FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0, ellipticity=1.0)

    def test_negative_ellipticity_rejected(self):
        with pytest.raises(ValueError, match="Ellipticity must be"):
            FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0, ellipticity=-0.1)


class TestFerrerComponentABC:
    def test_is_a_component(self):
        assert issubclass(FerrerComponent, Component)

    def test_implements_all_abstract_methods(self):
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0)
        ctx = _ctx()
        # Should not raise — all four abstract methods implemented
        spec = fc.to_libprofit_spec(ctx)
        derived = fc.derived_params(ctx)
        extent = fc.angular_extent_arcsec(ctx)
        assert isinstance(spec, str)
        assert isinstance(derived, dict)
        assert isinstance(extent, float)

    def test_axrat_property(self):
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0, ellipticity=0.5)
        assert fc.axrat == 0.5
        fc2 = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0)
        assert fc2.axrat == 1.0


class TestFerrerLibprofitSpec:
    def test_spec_includes_required_keys(self):
        fc = FerrerComponent(
            r_out_kpc=2.0, abs_mag=-18.0, alpha=2.0, beta=0.0,
            ellipticity=0.5, pa_deg=45,
        )
        spec = fc.to_libprofit_spec(_ctx())
        assert spec.startswith("ferrer:")
        for key in ("xcen=", "ycen=", "mag=", "rout=", "a=", "b=", "axrat=", "ang="):
            assert key in spec, f"missing key {key} in spec: {spec}"

    def test_spec_axrat_matches_ellipticity(self):
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0, ellipticity=0.7)
        spec = fc.to_libprofit_spec(_ctx())
        assert "axrat=0.30000000000000004" in spec or "axrat=0.3" in spec

    def test_spec_uses_apparent_mag_not_absolute(self):
        # M = -18, z = 0.01 ⇒ DM ≈ 33.18 ⇒ m_app ≈ 15.18
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0)
        ctx = _ctx(redshift=0.01)
        expected_app_mag = abs_to_app_mag(-18.0, 0.01)
        assert f"mag={expected_app_mag}" in fc.to_libprofit_spec(ctx)

    def test_spec_rout_in_pixels(self):
        # R_out=2 kpc at z=0.01 in 0.168 arcsec/pix
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0)
        ctx = _ctx(redshift=0.01, pixel_scale=0.168)
        expected_rout_pix = kpc_to_arcsec(2.0, 0.01) / 0.168
        spec = fc.to_libprofit_spec(ctx)
        assert f"rout={expected_rout_pix}" in spec


class TestFerrerAstropyHardError:
    def test_to_astropy_image_raises(self):
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0)
        with pytest.raises(NotImplementedError, match="libprofit"):
            fc.to_astropy_image(_ctx(), (51, 51))

    def test_generate_with_engine_astropy_raises_via_render_path(self):
        """Using engine='astropy' on a Ferrer galaxy should bubble up
        the NotImplementedError from to_astropy_image."""
        config = ImageConfig(size_pixels=51, engine="astropy")
        with pytest.raises(NotImplementedError, match="libprofit"):
            generate_mock_image(
                name="ferrer_only",
                redshift=0.01,
                components=[FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0)],
                config=config,
            )


class TestFerrerDerivedParams:
    def test_derived_params_keys(self):
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0, alpha=2.0, beta=0.0)
        ctx = _ctx()
        d = fc.derived_params(ctx)
        for k in ("profile", "r_out_arcsec", "r_out_pix", "app_mag",
                  "r_out_kpc", "abs_mag", "alpha", "beta",
                  "ellipticity", "pa_deg"):
            assert k in d, f"missing {k}"
        assert d["profile"] == "ferrer"
        assert d["r_out_kpc"] == 2.0
        assert d["abs_mag"] == -18.0

    def test_angular_extent_matches_kpc_conversion(self):
        fc = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0)
        ctx = _ctx(redshift=0.01)
        assert fc.angular_extent_arcsec(ctx) == pytest.approx(
            kpc_to_arcsec(2.0, 0.01)
        )


class TestFerrerRegistry:
    def test_built_from_dict(self):
        d = {
            "type": "ferrer",
            "r_out_kpc": 2.0,
            "abs_mag": -18.0,
            "alpha": 2.0,
            "beta": 0.0,
            "ellipticity": 0.4,
            "pa_deg": 30,
        }
        c = _build_component_from_dict(d)
        assert isinstance(c, FerrerComponent)
        assert c.r_out_kpc == 2.0
        assert c.ellipticity == 0.4

    def test_unknown_extra_keys_dropped(self):
        d = {
            "type": "ferrer",
            "r_out_kpc": 2.0,
            "abs_mag": -18.0,
            "irrelevant_key": "ignored",
        }
        # Should not raise; extra keys silently dropped (matches Sersic behavior).
        c = _build_component_from_dict(d)
        assert isinstance(c, FerrerComponent)


class TestFerrerYamlRoundtrip:
    def test_load_from_yaml_manifest(self, tmp_path):
        manifest = tmp_path / "ferrer_manifest.yaml"
        manifest.write_text(yaml.safe_dump({
            "galaxies": [{
                "name": "test_ferrer",
                "redshift": 0.01,
                "components": [
                    {
                        "type": "ferrer",
                        "id": "barfit",
                        "r_out_kpc": 2.0,
                        "abs_mag": -18.0,
                        "alpha": 2.0,
                        "beta": 0.0,
                        "ellipticity": 0.5,
                        "pa_deg": 45,
                    },
                ],
            }],
        }))
        galaxies = load_model_file(str(manifest))
        assert len(galaxies) == 1
        gal = galaxies[0]
        assert isinstance(gal, MockGalaxy)
        assert gal.name == "test_ferrer"
        assert len(gal.components) == 1
        assert isinstance(gal.components[0], FerrerComponent)
        assert gal.components[0].component_id == "barfit"
