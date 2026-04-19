#!/usr/bin/env python
"""
mockgal_galfit.py - Render mock galaxy models via the GALFIT binary.

Sidecar to ``mockgal.py`` that produces high-fidelity reference images
by writing a GALFIT-format `.ingal` file to a working directory and
invoking the GALFIT binary in model-only mode (`P) 1`). Used to:

1. Validate the libprofit-backed mockgal renderer against GALFIT itself
   (cross-correlation should be > 0.999 for sersic-only galaxies).
2. Render parsed S4G P4 outgal models pixel-perfectly for comparison
   against the original ``*_subcomps.fits`` cube hosted at IRSA.

Two input modes:

- **MockGalaxy mode**: pass a ``mockgal.MockGalaxy`` + ``ImageConfig``;
  the script builds an equivalent GALFIT block per Component using the
  ``mockgalaxy_to_galfit_dict`` adapter.
- **GALFIT-dict-direct mode**: pass a dict in the
  ``~/.claude/skills/galfit/scripts/galfit_io.py`` ``from_dict`` schema
  (header + components, all in image-plane units). Skips the MockGalaxy
  layer entirely — used to re-render parsed S4G outgals.

Usage:
    # API
    from mockgal_galfit import render_with_galfit
    img, meta = render_with_galfit(galaxy, config)

    # CLI (MockGalaxy mode)
    python mockgal_galfit.py --models inputs/huang2013/models/huang2013_models.yaml \
        --galaxy NGC1399 --output /tmp/galfit_render.fits

    # CLI (GALFIT-dict-direct mode; reads inputs/cs4g/cs4g_models.json)
    python mockgal_galfit.py --galfit-dict inputs/cs4g/cs4g_models.json \
        --galaxy NGC1097 --output /tmp/cs4g_NGC1097.fits
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits

# Import the mockgal core for MockGalaxy / Component handling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mockgal import (
    Component,
    FerrerComponent,
    ImageConfig,
    MockGalaxy,
    MockImageGenerator,
    PointSourceComponent,
    RenderContext,
    SersicComponent,
    abs_to_app_mag,
    kpc_to_arcsec,
    load_model_file,
)

# Make the bundled /galfit skill's parser importable.
_GALFIT_SKILL_PATH = Path.home() / ".claude" / "skills" / "galfit" / "scripts"
if str(_GALFIT_SKILL_PATH) not in sys.path:
    sys.path.insert(0, str(_GALFIT_SKILL_PATH))
import galfit_io  # type: ignore[import-not-found]


GALFIT_BINARY = os.environ.get("GALFIT_BIN", "/Users/shuang/code/galfit/galfit")


# ---------------------------------------------------------------------------
# Component -> GALFIT block conversion
# ---------------------------------------------------------------------------

def _to_fitted(value: float, fit: bool = False) -> Dict[str, Any]:
    """Build a FittedValue dict in the schema galfit_io.from_dict expects."""
    return {"value": float(value), "fit": bool(fit)}


def component_to_galfit_block(comp: Component, ctx: RenderContext) -> Dict[str, Any]:
    """Convert a mockgal Component to a galfit_io component dict.

    The output dict is keyed by attribute name (e.g. ``mag``, ``re``,
    ``n``, ``q``, ``pa``) matching the PARAM_SCHEMA of the corresponding
    galfit_io Component subclass. All FittedValue entries are marked
    ``fit=False`` because we're rendering, not fitting.

    Dispatches on Component subclass via isinstance. Future profile
    types extend this function with new branches.
    """
    if isinstance(comp, SersicComponent):
        d = comp.derived_params(ctx)
        return {
            "profile": "sersic",
            "x": _to_fitted(ctx.xcen_pix),
            "y": _to_fitted(ctx.ycen_pix),
            "mag": _to_fitted(d["app_mag"]),
            "re": _to_fitted(d["re_pix"]),
            "n": _to_fitted(comp.n),
            "q": _to_fitted(comp.axrat),
            "pa": _to_fitted(comp.pa_deg),
        }
    if isinstance(comp, FerrerComponent):
        d = comp.derived_params(ctx)
        return {
            "profile": "ferrer",
            "x": _to_fitted(ctx.xcen_pix),
            "y": _to_fitted(ctx.ycen_pix),
            "mu_central": _to_fitted(d["app_mag"]),  # NB: integrated mag passed as mu_central
            "r_out": _to_fitted(d["r_out_pix"]),
            "alpha": _to_fitted(comp.alpha),
            "beta": _to_fitted(comp.beta),
            "q": _to_fitted(comp.axrat),
            "pa": _to_fitted(comp.pa_deg),
        }
    if isinstance(comp, PointSourceComponent):
        dx, dy = comp._resolve_offset_pix(ctx)
        d = comp.derived_params(ctx)
        return {
            "profile": "psf",
            "x": _to_fitted(ctx.xcen_pix + dx),
            "y": _to_fitted(ctx.ycen_pix + dy),
            "mag": _to_fitted(d["app_mag"]),
        }
    raise TypeError(
        f"component_to_galfit_block: no GALFIT mapping for {type(comp).__name__}"
    )


def mockgalaxy_to_galfit_dict(
    galaxy: MockGalaxy,
    config: ImageConfig,
    image_shape: Tuple[int, int],
    input_image_path: str,
    output_block_path: str,
    psf_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a galfit_io.from_dict-compatible dict from a MockGalaxy.

    The image grid is set by ``image_shape`` and ``input_image_path``;
    the Component objects are translated via ``component_to_galfit_block``.
    PSF convolution uses the provided ``psf_image_path`` if not None.
    """
    ny, nx = image_shape
    ctx = RenderContext(
        redshift=galaxy.redshift,
        pixel_scale_arcsec_per_pix=config.pixel_scale,
        zeropoint=config.zeropoint,
        xcen_pix=nx / 2.0,
        ycen_pix=ny / 2.0,
    )
    components = [component_to_galfit_block(c, ctx) for c in galaxy.components]

    header: Dict[str, Any] = {
        "input_image": input_image_path,
        "output_block": output_block_path,
        "options": 1,                       # P) 1 = model only, no fitting
        "fit_region": [1, nx, 1, ny],       # H) full image
        "conv_box": [min(nx, 100), min(ny, 100)],   # I) convolution box
        "zeropoint": config.zeropoint,
        "plate_scale": [config.pixel_scale, config.pixel_scale],
        "display_type": "regular",
    }
    if psf_image_path is not None:
        header["psf_image"] = psf_image_path

    return {"header": header, "components": components, "leading_comments": []}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _resolve_image_shape(
    model: Union[MockGalaxy, dict, "galfit_io.GalfitFile"],
    config: ImageConfig,
) -> Tuple[int, int]:
    """Resolve the rendered image shape (ny, nx).

    For a MockGalaxy, defer to ``MockImageGenerator._compute_image_shape``
    (uses ``size_pixels`` or auto-sizes from the components' angular
    extent). For a galfit_io dict/GalfitFile, use the ``fit_region``
    (H header), falling back to ``config.size_pixels`` if absent.
    """
    if isinstance(model, MockGalaxy):
        gen = MockImageGenerator(config)
        params = gen._compute_derived_params(model)
        return gen._compute_image_shape(params)

    # dict or GalfitFile
    if isinstance(model, dict):
        fit_region = model.get("header", {}).get("fit_region")
    else:
        fit_region = model.header.fit_region  # type: ignore[union-attr]

    if fit_region is not None:
        xmin, xmax, ymin, ymax = fit_region
        return (int(ymax - ymin + 1), int(xmax - xmin + 1))

    if config.size_pixels is None:
        raise ValueError(
            "Cannot determine image shape: GALFIT model has no fit_region "
            "and ImageConfig.size_pixels is unset."
        )
    if isinstance(config.size_pixels, int):
        return (config.size_pixels, config.size_pixels)
    return (int(config.size_pixels[0]), int(config.size_pixels[1]))


def _write_zero_input_image(path: Path, shape: Tuple[int, int]) -> None:
    """Write a zeros input FITS so GALFIT picks up the grid dimensions."""
    data = np.zeros(shape, dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["EXPTIME"] = 1.0
    hdu.header["GAIN"] = 1.0
    hdu.header["NCOMBINE"] = 1
    hdu.writeto(path, overwrite=True)


def _write_psf_kernel(config: ImageConfig, path: Path) -> None:
    """Render the configured PSF kernel via the existing MockImageGenerator
    helper and write it to a FITS file (used by GALFIT's D) header)."""
    # MockImageGenerator._make_psf is the canonical PSF builder; reuse it
    # via a transient instance — does not run anything other than the PSF
    # construction path.
    if not config.psf_enabled:
        raise ValueError("PSF kernel requested but psf_enabled=False")
    transient = MockImageGenerator(config)
    psf = transient._make_psf()
    fits.writeto(path, psf.astype(np.float64), overwrite=True,
                 output_verify="silentfix")


def render_with_galfit(
    model: Union[MockGalaxy, dict, "galfit_io.GalfitFile"],
    config: ImageConfig,
    work_dir: Optional[Path] = None,
    keep_workdir: bool = False,
    galfit_bin: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render a model to a numpy array via the GALFIT binary.

    Parameters
    ----------
    model : MockGalaxy | dict | GalfitFile
        Either a mockgal MockGalaxy (translated via mockgalaxy_to_galfit_dict),
        or a galfit_io.from_dict-compatible dict, or a GalfitFile object.
    config : ImageConfig
        Provides pixel_scale, zeropoint, PSF settings, image shape (when
        the model doesn't carry its own fit_region).
    work_dir : Path, optional
        Working directory for transient files (input zeros, PSF, feedme,
        output). If None, a temp directory is created and cleaned up.
    keep_workdir : bool
        When True, do not remove ``work_dir`` after the render. Useful
        for debugging.
    galfit_bin : str, optional
        Override for the GALFIT binary path.

    Returns
    -------
    image : np.ndarray
        Rendered 2D image array.
    meta : dict
        Includes ``work_dir``, ``shape``, ``galfit_stdout``,
        ``galfit_stderr``, and the resolved feedme path.
    """
    binary = galfit_bin or GALFIT_BINARY
    if not Path(binary).exists():
        raise FileNotFoundError(f"GALFIT binary not found at {binary!r}")

    user_supplied_dir = work_dir is not None
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="mockgal_galfit_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        shape = _resolve_image_shape(model, config)
        zero_path = work_dir / "input_zero.fits"
        _write_zero_input_image(zero_path, shape)

        output_path = work_dir / "out.fits"
        psf_path: Optional[Path] = None
        if config.psf_enabled:
            psf_path = work_dir / "psf.fits"
            _write_psf_kernel(config, psf_path)

        # Build the GalfitFile
        if isinstance(model, MockGalaxy):
            gf_dict = mockgalaxy_to_galfit_dict(
                model, config, shape,
                input_image_path=zero_path.name,
                output_block_path=output_path.name,
                psf_image_path=(psf_path.name if psf_path else None),
            )
            gf = galfit_io.from_dict(gf_dict)
        elif isinstance(model, dict):
            # Coerce input/output paths to the local work_dir (caller may
            # have set them to absolute paths from elsewhere).
            patched = {
                "header": dict(model.get("header", {})),
                "components": list(model.get("components", [])),
                "leading_comments": list(model.get("leading_comments", [])),
            }
            patched["header"]["input_image"] = zero_path.name
            patched["header"]["output_block"] = output_path.name
            patched["header"]["options"] = 1
            if psf_path is not None:
                patched["header"]["psf_image"] = psf_path.name
            gf = galfit_io.from_dict(patched)
        else:
            gf = model
            gf.header.input_image = zero_path.name
            gf.header.output_block = output_path.name
            gf.header.options = 1
            if psf_path is not None:
                gf.header.psf_image = psf_path.name

        feedme_path = work_dir / "model.ingal"
        galfit_io.write_galfit(gf, feedme_path)

        result = subprocess.run(
            [binary, feedme_path.name],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=True,
        )

        if not output_path.exists():
            raise RuntimeError(
                f"GALFIT did not produce {output_path}. "
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        image = fits.getdata(output_path).astype(np.float64)

        meta: Dict[str, Any] = {
            "work_dir": str(work_dir),
            "shape": shape,
            "galfit_stdout": result.stdout,
            "galfit_stderr": result.stderr,
            "feedme": str(feedme_path),
            "output_fits": str(output_path),
        }
        return image, meta
    finally:
        if not keep_workdir and not user_supplied_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_galfit_dict_for_galaxy(path: Path, galaxy_name: str) -> dict:
    """Read a multi-galaxy GALFIT-dict JSON (e.g. cs4g_models.json) and
    return the entry for one galaxy in galfit_io.from_dict format.

    The CS4G ``cs4g_models.json`` schema is the dict-of-models produced
    by ``inputs/cs4g/scripts/parse_outgals.py`` — keyed by galaxy name,
    each value is { 'plate_scale_arcsec_per_pix', 'zeropoint',
    'fit_region', 'components': [...] }.
    """
    data = json.loads(path.read_text())
    if galaxy_name not in data:
        raise KeyError(
            f"Galaxy {galaxy_name!r} not in {path}. "
            f"Available: {sorted(data)[:10]}..."
        )
    entry = data[galaxy_name]
    # Assemble the header from the entry's flattened metadata.
    header: Dict[str, Any] = {}
    if entry.get("zeropoint") is not None:
        header["zeropoint"] = entry["zeropoint"]
    if entry.get("plate_scale_arcsec_per_pix") is not None:
        ps = entry["plate_scale_arcsec_per_pix"]
        header["plate_scale"] = [ps, ps]
    if entry.get("fit_region") is not None:
        header["fit_region"] = list(entry["fit_region"])
    return {
        "header": header,
        "components": entry.get("components", []),
        "leading_comments": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--models", type=Path,
                     help="Path to a mockgal MockGalaxy YAML/JSON manifest")
    src.add_argument("--galfit-dict", type=Path,
                     help="Path to a CS4G-style cs4g_models.json (dict-direct mode)")

    parser.add_argument("--galaxy", type=str, required=True,
                        help="Galaxy name to render")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the rendered FITS image")
    parser.add_argument("--config", type=Path,
                        help="Optional ImageConfig YAML/JSON (defaults applied otherwise)")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="Keep the GALFIT working dir (for debugging)")
    parser.add_argument("--galfit-bin", type=str, default=None,
                        help=f"Override GALFIT binary (default: $GALFIT_BIN or {GALFIT_BINARY})")
    args = parser.parse_args()

    # Build ImageConfig
    if args.config:
        from mockgal import load_image_configs
        cfgs = load_image_configs(str(args.config))
        if not cfgs:
            raise SystemExit(f"No configs found in {args.config}")
        config = cfgs[0]
    else:
        config = ImageConfig()

    if args.models is not None:
        galaxies = load_model_file(str(args.models), galaxy_names=[args.galaxy])
        if not galaxies:
            raise SystemExit(f"Galaxy {args.galaxy} not found in {args.models}")
        model: Union[MockGalaxy, dict] = galaxies[0]
    else:
        model = _load_galfit_dict_for_galaxy(args.galfit_dict, args.galaxy)

    image, meta = render_with_galfit(
        model, config,
        keep_workdir=args.keep_workdir,
        galfit_bin=args.galfit_bin,
    )
    fits.writeto(args.output, image, overwrite=True)
    print(f"Wrote {args.output}  shape={meta['shape']}  sum={image.sum():.6e}")
    if args.keep_workdir:
        print(f"Work dir kept: {meta['work_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
