"""
Unit tests for mockgal.py

Run with: pytest tests/test_mockgal.py -v
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import convolve as ndi_convolve

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import mockgal as mockgal_module

import mockgal as mockgal_module
from mockgal import (
    DEFAULT_PIXEL_SCALE,
    DEFAULT_REDSHIFT,
    DEFAULT_ZEROPOINT,
    HAS_MATPLOTLIB,
    MAX_SERSIC_INDEX,
    ImageConfig,
    MockGalaxy,
    MockImageGenerator,
    SersicComponent,
    SersicEngine,
    abs_to_app_mag,
    find_profit_cli,
    generate_mock_image,
    generate_mock_image_from_model,
    kpc_to_arcsec,
    load_model_file,
    parse_huang2013,
    save_fits,
    sb_limit_to_sigma,
    sb_mag_to_flux_per_pixel,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_component():
    """Single simple Sersic component."""
    return SersicComponent(
        r_eff_kpc=1.0,
        abs_mag=-20.0,
        n=4.0,
        ellipticity=0.0,
        pa_deg=0.0
    )


@pytest.fixture
def elliptical_component():
    """Elliptical Sersic component."""
    return SersicComponent(
        r_eff_kpc=2.0,
        abs_mag=-21.0,
        n=4.0,
        ellipticity=0.5,
        pa_deg=45.0
    )


@pytest.fixture
def simple_galaxy(simple_component):
    """Single-component galaxy."""
    return MockGalaxy(
        name="test_galaxy",
        redshift=0.01,
        components=[simple_component]
    )


@pytest.fixture
def multi_component_galaxy():
    """Multi-component galaxy."""
    return MockGalaxy(
        name="test_multi",
        redshift=0.01,
        components=[
            SersicComponent(r_eff_kpc=0.5, abs_mag=-18.0, n=1.0),
            SersicComponent(r_eff_kpc=2.0, abs_mag=-21.0, n=4.0),
        ]
    )


@pytest.fixture
def multi_component_galaxy_with_overall_re():
    """Multi-component galaxy with an explicit overall effective radius."""
    return MockGalaxy(
        name="test_multi_overall",
        redshift=0.01,
        re_overall=1.0,
        components=[
            SersicComponent(r_eff_kpc=0.5, abs_mag=-18.0, n=1.0),
            SersicComponent(r_eff_kpc=2.0, abs_mag=-21.0, n=4.0),
        ]
    )


@pytest.fixture
def default_config():
    """Default image configuration."""
    return ImageConfig()


@pytest.fixture
def huang_catalog_path():
    """Path to Huang 2013 catalog."""
    return Path(__file__).parent.parent / "inputs" / "huang2013_cgs_model.txt"


@pytest.fixture
def psf_image_path():
    """Path to a test PSF FITS file."""
    return Path(__file__).parent / "psf_gaussian.fits"


# =============================================================================
# Test Cosmology Functions
# =============================================================================

class TestCosmology:
    """Test cosmology conversion functions."""

    def test_kpc_to_arcsec_nearby(self):
        """At z=0.01, 1 kpc should be approximately 2.1 arcsec."""
        result = kpc_to_arcsec(1.0, z=0.01)
        # At z=0.01, D_A ~ 42 Mpc, so 1 kpc ~ 4.9 arcsec
        # (1 kpc / 42000 kpc) * 206265 arcsec/rad ~ 4.9 arcsec
        assert 4.0 < result < 6.0

    def test_kpc_to_arcsec_distant(self):
        """At higher z, angular size per kpc should decrease initially, then increase."""
        nearby = kpc_to_arcsec(1.0, z=0.01)
        medium = kpc_to_arcsec(1.0, z=0.1)
        # At small z, angular diameter distance increases with z
        # so angular size decreases
        assert medium < nearby

    def test_kpc_to_arcsec_scales_linearly(self):
        """Angular size should scale linearly with physical size."""
        r1 = kpc_to_arcsec(1.0, z=0.01)
        r2 = kpc_to_arcsec(2.0, z=0.01)
        assert np.isclose(r2, 2 * r1, rtol=1e-10)

    def test_abs_to_app_mag(self):
        """Check distance modulus calculation."""
        # At z=0.01, luminosity distance ~ 43 Mpc, dm ~ 33.2
        app = abs_to_app_mag(-20.0, z=0.01)
        assert 12 < app < 15  # -20 + ~33 = ~13

    def test_abs_to_app_mag_with_k_correction(self):
        """K-correction should add to apparent magnitude."""
        app1 = abs_to_app_mag(-20.0, z=0.01, k_corr=0.0)
        app2 = abs_to_app_mag(-20.0, z=0.01, k_corr=0.5)
        assert np.isclose(app2 - app1, 0.5, rtol=1e-10)


class TestSurfaceBrightnessConversions:
    """Test surface brightness conversion helpers."""

    def test_sb_mag_to_flux_per_pixel(self):
        """SB=ZP at 1 arcsec/pixel should give flux=1."""
        flux = sb_mag_to_flux_per_pixel(27.0, pixel_scale=1.0, zeropoint=27.0)
        assert np.isclose(flux, 1.0, rtol=1e-10)

    def test_sb_limit_to_sigma(self):
        """5-sigma SB limit converts to sigma=flux/5."""
        sigma = sb_limit_to_sigma(27.0, pixel_scale=1.0, zeropoint=27.0)
        assert np.isclose(sigma, 0.2, rtol=1e-10)


# =============================================================================
# Test Data Classes
# =============================================================================

class TestSersicComponent:
    """Test SersicComponent data class."""

    def test_valid_component(self, simple_component):
        """Valid component should be created without errors."""
        assert simple_component.n == 4.0
        assert simple_component.axrat == 1.0

    def test_axrat_property(self, elliptical_component):
        """Axis ratio should be 1 - ellipticity."""
        assert np.isclose(elliptical_component.axrat, 0.5)

    def test_invalid_negative_re(self):
        """Negative effective radius should raise error."""
        with pytest.raises(ValueError, match="r_eff_kpc must be positive"):
            SersicComponent(r_eff_kpc=-1.0, abs_mag=-20.0, n=4.0)

    def test_invalid_zero_re(self):
        """Zero effective radius should raise error."""
        with pytest.raises(ValueError, match="r_eff_kpc must be positive"):
            SersicComponent(r_eff_kpc=0.0, abs_mag=-20.0, n=4.0)

    def test_invalid_negative_sersic_n(self):
        """Negative Sersic index should raise error."""
        with pytest.raises(ValueError, match="Sersic index n must be positive"):
            SersicComponent(r_eff_kpc=1.0, abs_mag=-20.0, n=-1.0)

    def test_invalid_high_sersic_n(self):
        """Sersic index > 8 should raise error."""
        with pytest.raises(ValueError, match=f"Sersic index n must be <= {MAX_SERSIC_INDEX}"):
            SersicComponent(r_eff_kpc=1.0, abs_mag=-20.0, n=10.0)

    def test_boundary_sersic_n(self):
        """Sersic index exactly at boundary should work."""
        comp = SersicComponent(r_eff_kpc=1.0, abs_mag=-20.0, n=MAX_SERSIC_INDEX)
        assert comp.n == MAX_SERSIC_INDEX

    def test_invalid_ellipticity_too_high(self):
        """Ellipticity >= 1 should raise error."""
        with pytest.raises(ValueError, match="Ellipticity must be in"):
            SersicComponent(r_eff_kpc=1.0, abs_mag=-20.0, n=4.0, ellipticity=1.0)

    def test_invalid_ellipticity_negative(self):
        """Negative ellipticity should raise error."""
        with pytest.raises(ValueError, match="Ellipticity must be in"):
            SersicComponent(r_eff_kpc=1.0, abs_mag=-20.0, n=4.0, ellipticity=-0.1)


class TestMockGalaxy:
    """Test MockGalaxy data class."""

    def test_valid_galaxy(self, simple_galaxy):
        """Valid galaxy should be created without errors."""
        assert simple_galaxy.name == "test_galaxy"
        assert len(simple_galaxy.components) == 1

    def test_invalid_negative_redshift(self, simple_component):
        """Negative redshift should raise error."""
        with pytest.raises(ValueError, match="Redshift must be positive"):
            MockGalaxy(name="test", redshift=-0.01, components=[simple_component])

    def test_invalid_zero_redshift(self, simple_component):
        """Zero redshift should raise error."""
        with pytest.raises(ValueError, match="Redshift must be positive"):
            MockGalaxy(name="test", redshift=0.0, components=[simple_component])

    def test_invalid_empty_components(self):
        """Empty components should raise error."""
        with pytest.raises(ValueError, match="at least one component"):
            MockGalaxy(name="test", redshift=0.01, components=[])

    def test_total_abs_mag(self, multi_component_galaxy):
        """Total magnitude should be sum of component fluxes."""
        total = multi_component_galaxy.total_abs_mag
        # Should be brighter than any single component
        assert total < -18.0
        assert total < -21.0


class TestImageConfig:
    """Test ImageConfig data class."""

    def test_default_config(self, default_config):
        """Default config should have expected values."""
        assert default_config.pixel_scale == DEFAULT_PIXEL_SCALE
        assert default_config.zeropoint == DEFAULT_ZEROPOINT
        assert not default_config.psf_enabled
        assert not default_config.sky_enabled
        assert not default_config.noise_enabled

    def test_invalid_pixel_scale(self):
        """Invalid pixel scale should raise error."""
        with pytest.raises(ValueError, match="pixel_scale must be positive"):
            ImageConfig(pixel_scale=0.0)

    def test_invalid_psf_type(self):
        """Invalid PSF type should raise error."""
        with pytest.raises(ValueError, match="psf_type must be"):
            ImageConfig(psf_type="invalid")

    def test_invalid_engine(self):
        """Invalid engine should raise error."""
        with pytest.raises(ValueError, match="engine must be"):
            ImageConfig(engine="invalid")

    def test_valid_size_pixels_int(self):
        """Valid int size_pixels should work."""
        config = ImageConfig(size_pixels=100)
        assert config.size_pixels == 100

    def test_valid_size_pixels_tuple(self):
        """Valid tuple size_pixels should work."""
        config = ImageConfig(size_pixels=(100, 200))
        assert config.size_pixels == (100, 200)

    def test_valid_size_pixels_list_converted_to_tuple(self):
        """List size_pixels should be converted to tuple."""
        config = ImageConfig(size_pixels=[100, 200])
        assert config.size_pixels == (100, 200)
        assert isinstance(config.size_pixels, tuple)

    def test_invalid_size_pixels_negative(self):
        """Negative size_pixels should raise error."""
        with pytest.raises(ValueError, match="size_pixels must be positive"):
            ImageConfig(size_pixels=-100)

    def test_invalid_size_pixels_tuple_wrong_length(self):
        """size_pixels tuple with wrong length should raise error."""
        with pytest.raises(ValueError, match="size_pixels tuple must have 2 elements"):
            ImageConfig(size_pixels=(100, 200, 300))

    def test_invalid_size_pixels_tuple_negative(self):
        """size_pixels tuple with negative values should raise error."""
        with pytest.raises(ValueError, match="size_pixels elements must be positive"):
            ImageConfig(size_pixels=(100, -200))

    def test_invalid_size_pixels_type(self):
        """Invalid size_pixels type should raise error."""
        with pytest.raises(ValueError, match="size_pixels must be int or tuple"):
            ImageConfig(size_pixels="100")


# =============================================================================
# Test Sersic Engine
# =============================================================================

class TestSersicEngine:
    """Test Sersic rendering engine."""

    @pytest.fixture
    def engine(self):
        """Astropy engine (always available)."""
        return SersicEngine(engine="astropy")

    def test_engine_selection_auto(self):
        """Auto selection should work."""
        engine = SersicEngine(engine="auto")
        assert engine.engine in ("libprofit", "astropy")

    def test_engine_selection_astropy(self):
        """Astropy selection should work."""
        engine = SersicEngine(engine="astropy")
        assert engine.engine == "astropy"

    def test_engine_selection_auto_falls_back_when_profit_cli_unusable(self, monkeypatch):
        """Auto selection should ignore a profit-cli binary that cannot start."""
        mockgal_module._profit_cli_is_usable.cache_clear()
        monkeypatch.setattr(mockgal_module, "_resolve_profit_cli_path", lambda path: "/tmp/profit-cli")
        monkeypatch.setattr(mockgal_module, "_profit_cli_is_usable", lambda path: False)

        engine = SersicEngine(engine="auto", profit_cli_path="/tmp/profit-cli")

        assert engine.engine == "astropy"

    def test_render_circular_profile(self, engine):
        """Circular Sersic should be azimuthally symmetric."""
        image = engine.render(
            shape=(101, 101),
            xcen=50, ycen=50,
            mag=15.0, re_pix=20.0,
            n=4.0, axrat=1.0, ang=0.0,
            zeropoint=27.0
        )

        assert image.shape == (101, 101)
        assert image.max() > 0
        assert np.isfinite(image).all()

        # Check azimuthal symmetry
        # Pixel at (50, 70) should equal pixel at (50, 30) for circular profile
        assert np.allclose(image[50, 70], image[50, 30], rtol=0.05)
        assert np.allclose(image[70, 50], image[30, 50], rtol=0.05)

    def test_render_elliptical_profile(self, engine):
        """Elliptical profile should be elongated."""
        image = engine.render(
            shape=(101, 101),
            xcen=50, ycen=50,
            mag=15.0, re_pix=20.0,
            n=4.0, axrat=0.5, ang=0.0,  # PA=0 -> elongated along Y
            zeropoint=27.0
        )

        # At same distance from center, along major axis should be brighter
        # For PA=0 (from +Y), major axis is along Y
        # So (70, 50) should be brighter than (50, 70)
        assert image[70, 50] > image[50, 70]

    def test_render_high_sersic_index(self, engine):
        """High Sersic index should have peaked center."""
        image = engine.render(
            shape=(201, 201),
            xcen=100, ycen=100,
            mag=15.0, re_pix=30.0,
            n=8.0, axrat=1.0, ang=0.0,
            zeropoint=27.0
        )

        center_val = image[100, 100]
        edge_val = image[100, 130]  # at Re
        # High n should have very concentrated center
        assert center_val > 5 * edge_val

    def test_render_low_sersic_index(self, engine):
        """Low Sersic index (exponential) should be less concentrated."""
        image = engine.render(
            shape=(201, 201),
            xcen=100, ycen=100,
            mag=15.0, re_pix=30.0,
            n=1.0, axrat=1.0, ang=0.0,
            zeropoint=27.0
        )

        center_val = image[100, 100]
        edge_val = image[100, 130]  # at Re
        # n=1 (exponential) is less concentrated than n=4
        ratio = center_val / edge_val
        assert ratio < 20  # Much less peaked than n=4

    def test_render_high_ellipticity(self, engine):
        """High ellipticity (b/a < 0.3) should work."""
        image = engine.render(
            shape=(201, 201),
            xcen=100, ycen=100,
            mag=15.0, re_pix=30.0,
            n=1.0, axrat=0.25, ang=45.0,
            zeropoint=27.0
        )

        assert image.max() > 0
        assert np.isfinite(image).all()

    def test_flux_conservation(self, engine):
        """Total flux should match expected from magnitude."""
        mag = 15.0
        zp = 27.0
        expected_flux = 10 ** (-0.4 * (mag - zp))

        image = engine.render(
            shape=(501, 501),
            xcen=250, ycen=250,
            mag=mag, re_pix=50.0,
            n=4.0, axrat=1.0, ang=0.0,
            zeropoint=zp
        )

        total_flux = image.sum()
        # Allow 10% tolerance due to edge effects
        assert np.isclose(total_flux, expected_flux, rtol=0.1)

    def test_libprofit_psf_passes_P_argument(self, monkeypatch):
        """libprofit render should pass PSF FITS path via -P."""
        monkeypatch.setattr(mockgal_module, "find_profit_cli", lambda *_: "/fake/profit-cli")
        engine = SersicEngine(engine="libprofit")

        calls = []

        def _mock_run(cmd, capture_output, text, check):
            calls.append(cmd)
            return SimpleNamespace(stdout="1 2\n3 4\n", stderr="")

        monkeypatch.setattr(mockgal_module.subprocess, "run", _mock_run)

        psf = np.ones((3, 3), dtype=float)
        image = engine.render(
            shape=(2, 2),
            xcen=1.0,
            ycen=1.0,
            mag=15.0,
            re_pix=2.0,
            n=2.0,
            axrat=0.9,
            ang=30.0,
            zeropoint=27.0,
            psf=psf,
        )

        assert image.shape == (2, 2)
        assert calls
        assert "-P" in calls[0]

    def test_libprofit_psf_failure_falls_back_to_python_convolution(self, monkeypatch, caplog):
        """If -P path fails, engine should retry without -P and convolve in Python."""
        monkeypatch.setattr(mockgal_module, "find_profit_cli", lambda *_: "/fake/profit-cli")
        engine = SersicEngine(engine="libprofit")

        calls = []
        raw_image = np.array([[1.0, 2.0], [3.0, 4.0]])
        stdout = "\n".join(" ".join(str(v) for v in row) for row in raw_image)

        def _mock_run(cmd, capture_output, text, check):
            calls.append(cmd)
            if "-P" in cmd:
                raise subprocess.CalledProcessError(
                    returncode=1, cmd=cmd, stderr="psf read failed"
                )
            return SimpleNamespace(stdout=stdout, stderr="")

        monkeypatch.setattr(mockgal_module.subprocess, "run", _mock_run)

        psf = np.array(
            [[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )
        psf /= psf.sum()

        with caplog.at_level("WARNING"):
            image = engine.render(
                shape=(2, 2),
                xcen=1.0,
                ycen=1.0,
                mag=15.0,
                re_pix=2.0,
                n=2.0,
                axrat=0.9,
                ang=30.0,
                zeropoint=27.0,
                psf=psf,
            )

        assert len(calls) == 2
        assert "-P" in calls[0]
        assert "-P" not in calls[1]
        expected = ndi_convolve(raw_image, psf, mode="constant")
        assert np.allclose(image, expected)
        assert "falling back to Python convolution" in caplog.text


# =============================================================================
# Test Image Generator
# =============================================================================

class TestMockImageGenerator:
    """Test full image generation pipeline."""

    def test_generate_noiseless(self, simple_galaxy, default_config):
        """Generate noiseless image."""
        gen = MockImageGenerator(default_config)
        image, meta = gen.generate(simple_galaxy)

        assert image.ndim == 2
        assert image.max() > 0
        assert np.isfinite(image).all()
        assert 'components' in meta
        assert len(meta['components']) == 1

    def test_generate_with_psf(self, simple_galaxy):
        """Generate image with PSF convolution."""
        config = ImageConfig(psf_enabled=True, psf_fwhm=1.0)
        gen = MockImageGenerator(config)
        image, meta = gen.generate(simple_galaxy)

        assert image.max() > 0
        assert meta['psf_enabled'] is True
        assert meta['psf_type'] == 'gaussian'

    def test_generate_libprofit_path_avoids_post_convolution(self, simple_galaxy, monkeypatch):
        """libprofit path should pass PSF into engine and skip extra post-convolution."""
        config = ImageConfig(size_pixels=51, psf_enabled=True, psf_fwhm=1.0, engine="astropy")
        gen = MockImageGenerator(config)

        class DummyEngine:
            engine = "libprofit"

            def __init__(self):
                self.received_psf = None

            def render(self, **kwargs):
                self.received_psf = kwargs.get("psf")
                return np.ones(kwargs["shape"], dtype=np.float64)

        dummy_engine = DummyEngine()
        gen.engine = dummy_engine

        def _unexpected_convolve(*args, **kwargs):
            raise AssertionError("Post-convolution should not be called for libprofit path")

        monkeypatch.setattr(mockgal_module, "convolve", _unexpected_convolve)

        image, _ = gen.generate(simple_galaxy)
        assert image.shape == (51, 51)
        assert dummy_engine.received_psf is not None

    def test_psf_smoothing_effect(self, simple_galaxy):
        """PSF should smooth the image."""
        # Without PSF
        config_no_psf = ImageConfig(size_pixels=101, engine="astropy")
        gen_no_psf = MockImageGenerator(config_no_psf)
        image_no_psf, _ = gen_no_psf.generate(simple_galaxy)

        # With PSF
        config_psf = ImageConfig(size_pixels=101, psf_enabled=True, psf_fwhm=2.0, engine="astropy")
        gen_psf = MockImageGenerator(config_psf)
        image_psf, _ = gen_psf.generate(simple_galaxy)

        # PSF convolution should reduce peak value
        assert image_psf.max() < image_no_psf.max()

    def test_generate_with_psf_image_file(self, simple_galaxy, psf_image_path):
        """Generate image with PSF from a FITS file."""
        if not psf_image_path.exists():
            pytest.skip("Test PSF FITS file not found")

        config = ImageConfig(
            size_pixels=101,
            psf_enabled=True,
            psf_type="file",
            psf_file=str(psf_image_path),
        )
        gen = MockImageGenerator(config)
        image, meta = gen.generate(simple_galaxy)

        assert image.max() > 0
        assert meta["psf_enabled"] is True
        assert meta["psf_type"] == "file"

    def test_generate_with_flat_sky(self, simple_galaxy):
        """Generate image with flat sky background."""
        sky_level = 100.0
        config = ImageConfig(sky_enabled=True, sky_type="flat", sky_level=sky_level)
        gen = MockImageGenerator(config)
        image, meta = gen.generate(simple_galaxy)

        # Corners should be approximately sky level
        corner_val = image[0, 0]
        assert corner_val >= sky_level * 0.9  # Allow some galaxy flux

    def test_generate_with_tilted_sky(self, simple_galaxy):
        """Generate image with tilted sky background."""
        config = ImageConfig(
            sky_enabled=True,
            sky_type="tilted",
            sky_coeffs=[100.0, 0.1, 0.0, 0.0, 0.0, 0.0]  # Gradient in x
        )
        gen = MockImageGenerator(config)
        image, meta = gen.generate(simple_galaxy)

        # Right side should be brighter than left (positive x gradient)
        left = image[image.shape[0]//2, 0]
        right = image[image.shape[0]//2, -1]
        # Due to galaxy contribution, check relative difference
        assert right > left

    def test_generate_with_noise(self, simple_galaxy):
        """Generate image with Gaussian noise."""
        config = ImageConfig(noise_enabled=True, noise_sigma=10.0, noise_seed=42)
        gen = MockImageGenerator(config)

        image1, _ = gen.generate(simple_galaxy)
        image2, _ = gen.generate(simple_galaxy)

        # Same seed should give same noise
        np.testing.assert_array_equal(image1, image2)

    def test_noise_different_seeds(self, simple_galaxy):
        """Different seeds should give different noise."""
        config1 = ImageConfig(noise_enabled=True, noise_sigma=10.0, noise_seed=42)
        config2 = ImageConfig(noise_enabled=True, noise_sigma=10.0, noise_seed=43)

        gen1 = MockImageGenerator(config1)
        gen2 = MockImageGenerator(config2)

        image1, _ = gen1.generate(simple_galaxy)
        image2, _ = gen2.generate(simple_galaxy)

        # Different seeds should give different results
        assert not np.allclose(image1, image2)

    def test_generate_with_sky_sb_limit_noise(self, simple_galaxy):
        """Generate image with Gaussian noise from SB limit."""
        config = ImageConfig(
            size_pixels=101,
            noise_enabled=True,
            sky_sb_limit=27.0,
            noise_seed=42,
        )
        gen = MockImageGenerator(config)
        image, _ = gen.generate(simple_galaxy)

        assert image.max() > 0
        assert np.isfinite(image).all()

    def test_generate_with_sky_sb_value_poisson(self, simple_galaxy):
        """Generate image with Poisson noise from sky SB value."""
        config = ImageConfig(
            size_pixels=101,
            sky_enabled=True,
            sky_sb_value=22.0,
            noise_enabled=True,
            gain=4.0,
            noise_seed=42,
        )
        gen = MockImageGenerator(config)
        image1, _ = gen.generate(simple_galaxy)
        image2, _ = gen.generate(simple_galaxy)

        assert image1.max() > 0
        assert np.isfinite(image1).all()
        np.testing.assert_array_equal(image1, image2)

    def test_generate_multi_component(self, multi_component_galaxy, default_config):
        """Generate multi-component galaxy."""
        gen = MockImageGenerator(default_config)
        image, meta = gen.generate(multi_component_galaxy)

        assert 'components' in meta
        assert len(meta['components']) == 2

    def test_image_size_factor(self, simple_galaxy):
        """Image size should scale with size_factor."""
        config1 = ImageConfig(size_factor=10.0)
        config2 = ImageConfig(size_factor=20.0)

        gen1 = MockImageGenerator(config1)
        gen2 = MockImageGenerator(config2)

        image1, _ = gen1.generate(simple_galaxy)
        image2, _ = gen2.generate(simple_galaxy)

        # Larger factor should give larger image
        assert image2.shape[0] > image1.shape[0]

    def test_image_fixed_size(self, simple_galaxy):
        """Fixed size should override size_factor."""
        config = ImageConfig(size_pixels=201, size_factor=100.0)
        gen = MockImageGenerator(config)
        image, _ = gen.generate(simple_galaxy)

        assert image.shape == (201, 201)

    def test_image_rectangular_size(self, simple_galaxy):
        """Rectangular size should work with tuple."""
        config = ImageConfig(size_pixels=(150, 200))
        gen = MockImageGenerator(config)
        image, _ = gen.generate(simple_galaxy)

        assert image.shape == (150, 200)

    def test_end_to_end_output_and_engine(self, simple_galaxy, tmp_path):
        """Generate and save output, verify engine selection."""
        config = ImageConfig(size_pixels=51, engine="auto")
        gen = MockImageGenerator(config)
        image, meta = gen.generate(simple_galaxy)

        outpath = tmp_path / "test_galaxy.fits"
        save_fits(image, meta, outpath)

        assert outpath.exists()
        assert meta["engine"] in ("libprofit", "astropy")

        if find_profit_cli() is not None:
            assert meta["engine"] == "libprofit"

    def test_generate_mock_image_api(self):
        """Direct API should generate image and metadata."""
        image, meta = generate_mock_image(
            name="api_test",
            redshift=0.01,
            components=[
                {
                    "r_eff_kpc": 1.0,
                    "abs_mag": -20.0,
                    "n": 4.0,
                    "ellipticity": 0.2,
                    "pa_deg": 30.0,
                }
            ],
            config={"size_pixels": 51, "engine": "auto"},
        )

        assert image.shape == (51, 51)
        assert image.max() > 0
        assert meta["engine"] in ("libprofit", "astropy")

        if find_profit_cli() is not None:
            assert meta["engine"] == "libprofit"

    def test_generate_mock_image_from_model(self):
        """Generate image via model file helper."""
        model_path = Path(__file__).parent.parent / "examples" / "example_models.yaml"
        if not model_path.exists():
            pytest.skip("Model file not found")

        image, meta = generate_mock_image_from_model(
            model_path=model_path,
            galaxy_name="deV_galaxy",
            config={"size_pixels": 51, "engine": "auto"},
        )

        assert image.shape == (51, 51)
        assert image.max() > 0
        assert meta["name"] == "deV_galaxy"


class TestCLI:
    """CLI-level tests."""

    def test_cli_single_generates_output(self, tmp_path):
        """CLI should generate a FITS file in single mode."""
        repo_root = Path(__file__).parent.parent
        output_dir = tmp_path / "cli_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
        env["XDG_CACHE_HOME"] = str(tmp_path / "xdgcache")

        cmd = [
            sys.executable,
            "mockgal.py",
            "--single",
            "--name",
            "cli_test",
            "-z",
            "0.01",
            "--r-eff",
            "1.0",
            "--abs-mag",
            "-20.0",
            "--sersic-n",
            "4.0",
            "--size",
            "51",
            "--engine",
            "auto",
            "-o",
            str(output_dir),
        ]

        result = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert (output_dir / "cli_test.fits").exists()


# =============================================================================
# Test Huang Catalog Parser
# =============================================================================

class TestHuangParser:
    """Test parsing of Huang 2013 reference data."""

    def test_parse_catalog(self, huang_catalog_path):
        """Parse catalog should return galaxies."""
        if not huang_catalog_path.exists():
            pytest.skip("Catalog file not found")

        galaxies = parse_huang2013(str(huang_catalog_path))

        assert len(galaxies) > 0
        # Check a known galaxy exists
        assert any("NGC" in name for name in galaxies.keys())

    def test_component_count(self, huang_catalog_path):
        """Galaxies should have multiple components."""
        if not huang_catalog_path.exists():
            pytest.skip("Catalog file not found")

        galaxies = parse_huang2013(str(huang_catalog_path))

        # Find a galaxy with 3 components
        multi_comp = [name for name, comps in galaxies.items() if len(comps) >= 3]
        assert len(multi_comp) > 0

    def test_component_parameters(self, huang_catalog_path):
        """Component parameters should be valid."""
        if not huang_catalog_path.exists():
            pytest.skip("Catalog file not found")

        galaxies = parse_huang2013(str(huang_catalog_path))

        for name, components in galaxies.items():
            for comp in components:
                assert comp.r_eff_kpc > 0
                assert 0 < comp.n <= MAX_SERSIC_INDEX
                assert 0 <= comp.ellipticity < 1

    def test_known_galaxy_ic1459(self, huang_catalog_path):
        """IC 1459 should have expected structure."""
        if not huang_catalog_path.exists():
            pytest.skip("Catalog file not found")

        galaxies = parse_huang2013(str(huang_catalog_path))

        if "IC 1459" not in galaxies:
            pytest.skip("IC 1459 not in catalog")

        ic1459 = galaxies["IC 1459"]
        assert len(ic1459) == 3  # 3-component model
        # First component should have smallest Re
        assert ic1459[0].r_eff_kpc < ic1459[1].r_eff_kpc

    def test_load_model_file_reads_re_overall(self):
        """Model loader should preserve galaxy-level re_overall metadata."""
        model_path = Path(__file__).parent.parent / "inputs" / "huang2013" / "models" / "huang2013_models.yaml"
        galaxies = load_model_file(str(model_path), galaxy_names=["IC 1459"])

        assert len(galaxies) == 1
        assert galaxies[0].re_overall is not None
        assert galaxies[0].re_overall > 0


# =============================================================================
# Test Output Functions
# =============================================================================

class TestOutput:
    """Test output saving functions."""

    def test_save_fits(self, simple_galaxy, default_config, tmp_path):
        """Save FITS file with metadata."""
        gen = MockImageGenerator(default_config)
        image, meta = gen.generate(simple_galaxy)

        outpath = tmp_path / "test.fits"
        save_fits(image, meta, outpath)

        assert outpath.exists()

        # Read back and verify
        from astropy.io import fits
        with fits.open(outpath) as hdul:
            assert hdul[0].data.shape == image.shape
            header = hdul[0].header
            assert header['OBJECT'] == simple_galaxy.name
            assert header['NCOMP'] == 1

    def test_output_png_without_fits(self, simple_galaxy, default_config, tmp_path):
        """Test generating PNG directly without saving FITS."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not available")

        from mockgal import visualize_galaxy

        gen = MockImageGenerator(default_config)
        image, metadata = gen.generate(simple_galaxy)

        # Generate PNG directly without FITS
        png_path = tmp_path / "direct.png"
        visualize_galaxy(image, metadata=metadata, output_path=str(png_path))

        assert png_path.exists()
        assert not (tmp_path / "direct.fits").exists()

    def test_filename_convention(self):
        """Test new filename convention (no underscores in galaxy names)."""
        from mockgal import sanitize_filename

        galaxy_name = "NGC 3923"
        config_name = "clean"

        safe_galaxy = sanitize_filename(galaxy_name)
        safe_config = sanitize_filename(config_name)

        filename = f"{safe_galaxy}_{safe_config}.fits"
        assert filename == "NGC3923_clean.fits"
        assert "_" in filename  # Underscore between galaxy and config
        assert filename.count("_") == 1  # Only one underscore


# =============================================================================
# Test Visualization
# =============================================================================

class TestVisualization:
    """Tests for visualization functions."""

    def test_visualize_from_fits(self, simple_galaxy, default_config, tmp_path):
        """Test visualizing a FITS file (auto-generates PNG)."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not available")

        from mockgal import visualize_galaxy

        gen = MockImageGenerator(default_config)
        image, metadata = gen.generate(simple_galaxy)

        fits_path = tmp_path / "test.fits"
        png_path = tmp_path / "test.png"
        save_fits(image, metadata, fits_path)

        # Should auto-generate PNG with same prefix
        visualize_galaxy(str(fits_path))
        assert png_path.exists()

    def test_visualize_from_array(self, simple_galaxy, default_config, tmp_path):
        """Test visualizing from numpy array with metadata."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not available")

        from mockgal import visualize_galaxy

        gen = MockImageGenerator(default_config)
        image, metadata = gen.generate(simple_galaxy)

        output_path = tmp_path / "test_array.png"
        visualize_galaxy(image, metadata=metadata, output_path=str(output_path))
        assert output_path.exists()

    def test_visualize_custom_colormap(self, simple_galaxy, default_config, tmp_path):
        """Test visualization with custom colormap."""
        if not HAS_MATPLOTLIB:
            pytest.skip("matplotlib not available")

        from mockgal import visualize_galaxy

        gen = MockImageGenerator(default_config)
        image, metadata = gen.generate(simple_galaxy)

        output_path = tmp_path / "test_magma.png"
        visualize_galaxy(image, metadata=metadata, output_path=str(output_path), cmap='magma')
        assert output_path.exists()


# =============================================================================
# Test Convention Conversions
# =============================================================================

class TestConventions:
    """Test parameter convention conversions between engines."""

    def test_pa_convention(self):
        """Position angle convention: pyprofit from +Y, astropy from +X."""
        # pyprofit PA=0 means along +Y
        # astropy theta=0 means along +X, theta=90deg means along +Y
        pa_pyprofit = 0.0  # Along +Y
        theta_astropy = np.radians(90 - pa_pyprofit)
        assert np.isclose(theta_astropy, np.pi/2)

        pa_pyprofit = 90.0  # Along +X
        theta_astropy = np.radians(90 - pa_pyprofit)
        assert np.isclose(theta_astropy, 0.0)

    def test_ellipticity_convention(self):
        """Ellipticity convention: pyprofit uses axrat, astropy uses ellip."""
        axrat = 0.7  # b/a
        ellip = 1 - axrat
        assert np.isclose(ellip, 0.3)

        # Round-trip
        axrat_recovered = 1 - ellip
        assert np.isclose(axrat_recovered, axrat)

    def test_sanitize_filename(self):
        """Test filename sanitization removes spaces, not replaces with underscores."""
        from mockgal import sanitize_filename

        assert sanitize_filename("NGC 3923") == "NGC3923"
        assert sanitize_filename("IC 1459") == "IC1459"
        assert sanitize_filename("ESO 185-G054") == "ESO185-G054"
        assert sanitize_filename("NGC  1399") == "NGC1399"  # multiple spaces
        assert sanitize_filename("simple") == "simple"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
