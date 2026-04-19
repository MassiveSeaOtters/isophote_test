"""
Integration test: run a real GALFIT binary against a parser-written input
and parse the resulting restart file.

The test is skipped automatically if the expected binary is not installed
at GALFIT_BIN.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import galfit_io as gio


GALFIT_BIN = Path("/Users/shuang/code/galfit/galfit")
PSF_FIXTURE = Path(__file__).parent / "psf_gaussian.fits"


pytestmark = pytest.mark.skipif(
    not GALFIT_BIN.exists(), reason="GALFIT binary not installed at expected path",
)


def _make_synthetic_data(path: Path, psf_path: Path, size: int = 80) -> None:
    """Render a Sersic galaxy onto a square FITS image with mild noise."""
    import numpy as np
    from astropy.io import fits
    from astropy.modeling.models import Sersic2D
    from scipy.ndimage import convolve

    model = Sersic2D(
        amplitude=100.0, r_eff=6.0, n=2.0,
        x_0=size / 2, y_0=size / 2, ellip=0.3, theta=np.radians(30.0),
    )
    y, x = np.mgrid[:size, :size]
    img = np.asarray(model(x, y), dtype=float)

    psf = fits.getdata(str(psf_path)).astype(float)
    psf = psf / psf.sum()
    img = convolve(img, psf, mode="constant")
    rng = np.random.default_rng(42)
    img = img + rng.normal(scale=max(float(img.max()) * 1e-3, 1e-6), size=img.shape)

    hdu = fits.PrimaryHDU(img.astype("float32"))
    hdu.header["EXPTIME"] = 1.0
    hdu.header["GAIN"] = 1.0
    hdu.header["NCOMBINE"] = 1
    hdu.header["RDNOISE"] = 1.0
    hdu.writeto(path, overwrite=True)


def _build_config(
    data: Path, psf: Path, size: int,
) -> gio.GalfitFile:
    header = gio.GalfitHeader(
        input_image=data.name,
        output_block="imgblock.fits",
        sigma_image="none",
        psf_image=psf.name,
        psf_fine_sampling=1,
        bad_pixel_mask="none",
        constraint_file="none",
        fit_region=(1, size, 1, size),
        conv_box=(30, 30),
        zeropoint=27.0,
        plate_scale=(0.1, 0.1),
        display_type="regular",
        options=0,
    )
    sersic = gio.SersicComponent(
        x=gio.FittedValue(size / 2),
        y=gio.FittedValue(size / 2),
        mag=gio.FittedValue(20.0),
        re=gio.FittedValue(5.0),
        n=gio.FittedValue(2.0),
        q=gio.FittedValue(0.8),
        pa=gio.FittedValue(30.0),
    )
    sky = gio.SkyComponent(
        dc=gio.FittedValue(0.0, fit=False),
        dsky_dx=gio.FittedValue(0.0, fit=False),
        dsky_dy=gio.FittedValue(0.0, fit=False),
    )
    return gio.GalfitFile(header=header, components=[sersic, sky])


def test_galfit_accepts_parser_written_config(tmp_path: Path):
    if not PSF_FIXTURE.exists():
        pytest.skip("PSF fixture missing")
    size = 80
    data = tmp_path / "data.fits"
    psf = tmp_path / "psf.fits"
    shutil.copy(PSF_FIXTURE, psf)
    _make_synthetic_data(data, psf, size=size)

    gf = _build_config(data, psf, size)
    feedme = tmp_path / "feedme.galfit"
    gio.write_galfit(gf, feedme)

    # Run GALFIT with a 1-iteration cap so the test completes in seconds.
    result = subprocess.run(
        [str(GALFIT_BIN), "-imax", "1", feedme.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # Exit code 0 means GALFIT produced a solution; 1 means it exited
    # without one. Either is acceptable here: what we care about is that
    # the config parsed and GALFIT attempted the fit without an error.
    combined = (result.stdout + result.stderr).lower()
    assert "input menu file" not in combined or "doh" not in combined, (
        f"GALFIT stderr={result.stderr!r}"
    )
    assert "bad format" not in combined, f"GALFIT stderr={result.stderr!r}"
    # GALFIT should have written galfit.01 (restart) and imgblock.fits.
    restart = tmp_path / "galfit.01"
    assert restart.exists(), (
        f"no galfit.01 in {list(tmp_path.iterdir())}, stderr={result.stderr!r}"
    )


def test_parser_reads_galfit_restart_output(tmp_path: Path):
    """After GALFIT runs, parse its galfit.01 output."""
    if not PSF_FIXTURE.exists():
        pytest.skip("PSF fixture missing")
    size = 80
    data = tmp_path / "data.fits"
    psf = tmp_path / "psf.fits"
    shutil.copy(PSF_FIXTURE, psf)
    _make_synthetic_data(data, psf, size=size)

    gf = _build_config(data, psf, size)
    feedme = tmp_path / "feedme.galfit"
    gio.write_galfit(gf, feedme)

    subprocess.run(
        [str(GALFIT_BIN), "-imax", "2", feedme.name],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    restart = tmp_path / "galfit.01"
    if not restart.exists():
        pytest.skip(f"GALFIT did not produce a restart file for this input")

    parsed = gio.read_galfit(restart)
    assert parsed.header.input_image == data.name
    assert parsed.header.fit_region == (1, size, 1, size)
    # At least a sersic and a sky component.
    profiles = [type(c).__name__ for c in parsed.components]
    assert "SersicComponent" in profiles
    assert "SkyComponent" in profiles

    # The Sersic in the restart file should carry the initial-guess
    # values we wrote, possibly refined. Uncertainty columns only appear
    # once GALFIT has computed a covariance matrix, which needs enough
    # iterations to converge - skip checking them here and cover that
    # path via the output_decorations.galfit fixture instead.
    sersic = next(c for c in parsed.components
                  if isinstance(c, gio.SersicComponent))
    assert sersic.mag.value > 0
    assert sersic.re.value > 0
    # GALFIT 3.0.7 also emits reserved placeholder slots 6) 7) 8) on
    # sersic; those must land in extra_params, not cause a parse error.
    assert set(sersic.extra_params.keys()) >= {6, 7, 8}
