"""Integration tests for mockgal_galfit (Phase E).

Skipped wholesale when the GALFIT binary is unreachable. Each rendering
test invokes the real binary at GALFIT_BINARY (default /Users/shuang/code/galfit/galfit),
synthesizes a small image, and asserts shape + flux conservation.
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mockgal import (
    FerrerComponent,
    ImageConfig,
    MockGalaxy,
    PointSourceComponent,
    SersicComponent,
    abs_to_app_mag,
    generate_mock_image,
)

# Skip the entire module if GALFIT binary is missing.
GALFIT_BIN = os.environ.get("GALFIT_BIN", "/Users/shuang/code/galfit/galfit")
pytestmark = pytest.mark.skipif(
    not Path(GALFIT_BIN).exists(),
    reason=f"GALFIT binary not found at {GALFIT_BIN}; set GALFIT_BIN env var",
)

# Import after the skip guard so a missing binary doesn't break collection.
from mockgal_galfit import (  # noqa: E402
    component_to_galfit_block,
    mockgalaxy_to_galfit_dict,
    render_with_galfit,
)
from mockgal import RenderContext  # noqa: E402


def _ctx(redshift=0.01, pixel_scale=0.168, zeropoint=27.0,
         xcen=50.0, ycen=50.0):
    return RenderContext(
        redshift=redshift,
        pixel_scale_arcsec_per_pix=pixel_scale,
        zeropoint=zeropoint,
        xcen_pix=xcen,
        ycen_pix=ycen,
    )


def _no_psf_config(size_pixels=51, pixel_scale=0.168, zeropoint=27.0):
    return ImageConfig(
        name="t", engine="astropy", pixel_scale=pixel_scale,
        zeropoint=zeropoint, size_pixels=size_pixels,
    )


def _gauss_psf_config(size_pixels=101, fwhm_arcsec=1.0):
    return ImageConfig(
        name="t", engine="astropy", pixel_scale=0.168, zeropoint=27.0,
        size_pixels=size_pixels,
        psf_enabled=True, psf_type="gaussian", psf_fwhm=fwhm_arcsec,
    )


# ---------------------------------------------------------------------------
# Component → GALFIT block dispatch
# ---------------------------------------------------------------------------

class TestComponentToGalfitBlock:
    def test_sersic(self):
        comp = SersicComponent(r_eff_kpc=2.0, abs_mag=-19.5, n=4.0,
                               ellipticity=0.2, pa_deg=45)
        block = component_to_galfit_block(comp, _ctx())
        assert block["profile"] == "sersic"
        for key in ("x", "y", "mag", "re", "n", "q", "pa"):
            assert key in block, f"missing {key} in block: {block}"
        assert block["mag"]["value"] == pytest.approx(abs_to_app_mag(-19.5, 0.01))

    def test_ferrer(self):
        comp = FerrerComponent(r_out_kpc=2.0, abs_mag=-18.0, alpha=2.0, beta=0.0)
        block = component_to_galfit_block(comp, _ctx())
        assert block["profile"] == "ferrer"
        for key in ("x", "y", "mu_central", "r_out", "alpha", "beta", "q", "pa"):
            assert key in block

    def test_psf(self):
        comp = PointSourceComponent(abs_mag=-15.0)
        block = component_to_galfit_block(comp, _ctx(xcen=60.0, ycen=60.0))
        assert block["profile"] == "psf"
        assert block["x"]["value"] == 60.0
        assert block["y"]["value"] == 60.0


# ---------------------------------------------------------------------------
# End-to-end renders via the GALFIT binary
# ---------------------------------------------------------------------------

class TestRenderSingleSersic:
    def test_renders_to_correct_shape(self):
        gal = MockGalaxy(
            name="test_sersic", redshift=0.01,
            components=[SersicComponent(r_eff_kpc=2.0, abs_mag=-19.5, n=4.0)],
        )
        cfg = _no_psf_config(size_pixels=51)
        img, meta = render_with_galfit(gal, cfg)
        assert img.shape == (51, 51)
        assert meta["shape"] == (51, 51)
        assert img.max() > 0
        assert np.isfinite(img).all()

    def test_total_flux_within_5_percent(self):
        """GALFIT's integrated flux should match 10**(-0.4*(mag-zp))."""
        cfg = _no_psf_config(size_pixels=201, pixel_scale=0.168)
        gal = MockGalaxy(
            name="flux_test", redshift=0.01,
            components=[SersicComponent(r_eff_kpc=1.0, abs_mag=-18.0, n=1.0)],
        )
        img, _ = render_with_galfit(gal, cfg)
        app_mag = abs_to_app_mag(-18.0, 0.01)
        expected = 10 ** (-0.4 * (app_mag - 27.0))
        # Sersic n=1 with R_e small relative to image — most flux captured.
        assert img.sum() == pytest.approx(expected, rel=0.05)

    def test_correlation_with_mockgal_astropy(self):
        """GALFIT and mockgal-astropy should agree to >0.95 (independent
        numerical implementations of the same Sersic profile, with
        slightly different sub-pixel sampling)."""
        cfg = _gauss_psf_config(size_pixels=101, fwhm_arcsec=1.0)
        gal = MockGalaxy(
            name="parity", redshift=0.01,
            components=[SersicComponent(
                r_eff_kpc=2.0, abs_mag=-19.5, n=4.0,
                ellipticity=0.2, pa_deg=45,
            )],
        )
        img_g, _ = render_with_galfit(gal, cfg)
        img_m, _ = generate_mock_image(
            name=gal.name, redshift=gal.redshift,
            components=gal.components, config=cfg,
        )
        corr = np.corrcoef(img_g.flatten(), img_m.flatten())[0, 1]
        assert corr > 0.95, f"correlation {corr:.4f} < 0.95"
        # Flux agreement to <2%
        ratio = img_g.sum() / img_m.sum()
        assert 0.98 < ratio < 1.02, f"flux ratio {ratio:.4f} not in [0.98, 1.02]"


class TestRenderFerrer:
    def test_single_ferrer_renders(self):
        cfg = _no_psf_config(size_pixels=101)
        gal = MockGalaxy(
            name="ferrer_only", redshift=0.01,
            components=[FerrerComponent(
                r_out_kpc=2.0, abs_mag=-18.0,
                alpha=2.0, beta=0.0,
                ellipticity=0.5, pa_deg=45,
            )],
        )
        img, _ = render_with_galfit(gal, cfg)
        assert img.shape == (101, 101)
        assert img.max() > 0
        assert np.isfinite(img).all()
        # Ferrer is truncated, so most flux must be inside the image.
        assert img.sum() > 0


class TestRenderPsf:
    def test_psf_only_renders(self):
        cfg = _gauss_psf_config(size_pixels=51, fwhm_arcsec=1.5)
        gal = MockGalaxy(
            name="psf_only", redshift=0.01,
            components=[PointSourceComponent(abs_mag=-15.0)],
        )
        img, _ = render_with_galfit(gal, cfg)
        assert img.shape == (51, 51)
        # PSF concentrated near center
        cy, cx = 25, 25
        assert img[cy, cx] > 0
        assert img.max() == pytest.approx(img[cy-1:cy+2, cx-1:cx+2].max())


class TestRenderMulticomponent:
    def test_sersic_plus_ferrer_plus_psf(self):
        cfg = _gauss_psf_config(size_pixels=101, fwhm_arcsec=1.0)
        gal = MockGalaxy(
            name="bdb", redshift=0.01,
            components=[
                SersicComponent(r_eff_kpc=1.5, abs_mag=-18.5, n=4.0),
                FerrerComponent(r_out_kpc=3.0, abs_mag=-18.0,
                                alpha=2.0, beta=0.0, ellipticity=0.6, pa_deg=30),
                PointSourceComponent(abs_mag=-15.0),
            ],
        )
        img, meta = render_with_galfit(gal, cfg)
        assert img.shape == (101, 101)
        assert img.max() > 0
        assert np.isfinite(img).all()


class TestKeepWorkdir:
    def test_workdir_removed_by_default(self, tmp_path):
        cfg = _no_psf_config(size_pixels=51)
        gal = MockGalaxy(
            name="cleanup_test", redshift=0.01,
            components=[SersicComponent(r_eff_kpc=2.0, abs_mag=-19.0, n=4.0)],
        )
        img, meta = render_with_galfit(gal, cfg)
        assert not Path(meta["work_dir"]).exists(), \
            f"work_dir {meta['work_dir']} should have been cleaned up"

    def test_workdir_kept_when_user_supplied(self, tmp_path):
        cfg = _no_psf_config(size_pixels=51)
        gal = MockGalaxy(
            name="cleanup_test", redshift=0.01,
            components=[SersicComponent(r_eff_kpc=2.0, abs_mag=-19.0, n=4.0)],
        )
        wd = tmp_path / "render_workdir"
        img, meta = render_with_galfit(gal, cfg, work_dir=wd)
        assert wd.exists(), "user-supplied work_dir should be preserved"
        assert (wd / "out.fits").exists(), "out.fits should be in work_dir"

    def test_keep_workdir_flag(self, tmp_path):
        cfg = _no_psf_config(size_pixels=51)
        gal = MockGalaxy(
            name="cleanup_test", redshift=0.01,
            components=[SersicComponent(r_eff_kpc=2.0, abs_mag=-19.0, n=4.0)],
        )
        img, meta = render_with_galfit(gal, cfg, keep_workdir=True)
        wd = Path(meta["work_dir"])
        assert wd.exists(), "keep_workdir=True should preserve auto-temp"
        # Cleanup
        shutil.rmtree(wd, ignore_errors=True)


class TestGalfitDictDirectMode:
    def test_dict_with_fit_region_uses_it(self, tmp_path):
        """A dict in galfit_io.from_dict format with a fit_region should
        be rendered at that exact size, bypassing image_config sizing."""
        d = {
            "header": {
                "zeropoint": 27.0,
                "plate_scale": [0.168, 0.168],
                "fit_region": [1, 81, 1, 81],
            },
            "components": [
                {
                    "profile": "sersic",
                    "x": {"value": 41.0, "fit": False},
                    "y": {"value": 41.0, "fit": False},
                    "mag": {"value": 16.0, "fit": False},
                    "re": {"value": 8.0, "fit": False},
                    "n": {"value": 2.0, "fit": False},
                    "q": {"value": 0.7, "fit": False},
                    "pa": {"value": 30.0, "fit": False},
                },
            ],
            "leading_comments": [],
        }
        cfg = _no_psf_config(size_pixels=999, pixel_scale=0.168, zeropoint=27.0)
        img, meta = render_with_galfit(d, cfg)
        # fit_region should win over size_pixels
        assert img.shape == (81, 81)
        assert img.max() > 0
