#!/usr/bin/env python
"""
qa_mockgal_vs_s4g.py - P8.9 validation of mockgal renders against the
original Salo+2015 S4G `*_subcomps.fits` cubes.

Pipeline:
  1. Read the CS4G model YAML (inputs/cs4g/models/cs4g_sample_models.yaml).
  2. For each target galaxy, build a MockGalaxy and render it via mockgal
     libprofit at Spitzer/IRAC1 native settings (0.75"/pix, zp=21.097).
  3. Load the pre-fetched S4G subcomps cube, sum the component planes to
     build the noise-free reference image, and crop both images around
     their respective galaxy centers to a common FOV.
  4. Compute flux ratio, peak ratio, correlation, residual RMS, and a
     mean radial profile of each side.
  5. Write per-galaxy mockgal.fits + s4g_reference.fits + the comparison
     PNG (output/cs4g_s4g_irac1_test/qa_s4g_validation.png) + a summary
     JSON (output/cs4g_s4g_irac1_test/qa_summary.json).

Assumes the S4G subcomps cubes live at
output/cs4g_s4g_irac1_test/{name}/s4g_subcomps.fits (already fetched from
the IRSA P4 directory; see the session handover doc).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# libprofit's profit-cli has a hardcoded @rpath to /Users/mac/...; make
# the local build dir visible before importing mockgal.
LIBPROFIT_DIR = REPO_ROOT / "libprofit" / "build"
os.environ.setdefault("LIBPROFIT_PATH", str(LIBPROFIT_DIR))
_extra = str(LIBPROFIT_DIR)
_existing = os.environ.get("DYLD_LIBRARY_PATH", "")
if _extra not in _existing.split(":"):
    os.environ["DYLD_LIBRARY_PATH"] = f"{_extra}:{_existing}" if _existing else _extra

from mockgal import (  # noqa: E402
    FerrerComponent,
    ImageConfig,
    MockGalaxy,
    PointSourceComponent,
    SersicComponent,
    generate_mock_image,
)

MODEL_YAML = REPO_ROOT / "inputs" / "cs4g" / "models" / "cs4g_sample_models.yaml"
OUT_ROOT = REPO_ROOT / "output" / "cs4g_s4g_irac1_test"

# S4G / manifest settings (mirror inputs/cs4g/runs/cs4g_s4g_irac1_test.yaml).
PIXEL_SCALE = 0.75
ZEROPOINT = 21.097
PSF_FWHM = 1.66
PSF_TYPE = "gaussian"
SIZE_FACTOR = 6.0
MAX_IMAGE_SIZE = 4001

TARGETS = ["NGC1433", "NGC0275", "NGC1357"]


# --------------------------------------------------------------------------- #
# MockGalaxy construction
# --------------------------------------------------------------------------- #

def load_model_dict(path: Path) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text())
    return {g["name"]: g for g in data["galaxies"]}


def build_mockgalaxy(name: str, record: dict) -> MockGalaxy:
    # The CS4G sample YAML stores abs_mag in predicted i-band (see the
    # metadata's `assumed_band` field). For P8.9 we compare against the
    # S4G IRAC1 3.6 um subcomps cubes using zp=21.097, so we add the
    # per-galaxy color term (M_3.6 - M_i = color_shift_3p6_minus_i,
    # positive because IRAC1 is brighter for these late-type disks) to
    # each component's abs_mag to recover the 3.6 um magnitude.
    color_shift = float(record.get("color_shift_3p6_minus_i", 0.0))
    comps = []
    for c in record["components"]:
        t = c["type"]
        cid = c.get("id")
        abs_mag_3p6 = float(c["abs_mag"]) + color_shift
        if t == "sersic":
            comps.append(SersicComponent(
                r_eff_kpc=c["r_eff_kpc"], abs_mag=abs_mag_3p6, n=c["n"],
                ellipticity=c["ellipticity"], pa_deg=c["pa_deg"],
                component_id=cid,
            ))
        elif t == "ferrer":
            comps.append(FerrerComponent(
                r_out_kpc=c["r_out_kpc"], abs_mag=abs_mag_3p6,
                alpha=c["alpha"], beta=c["beta"],
                ellipticity=c["ellipticity"], pa_deg=c["pa_deg"],
                component_id=cid,
            ))
        elif t == "psf":
            comps.append(PointSourceComponent(abs_mag=abs_mag_3p6, component_id=cid))
        else:
            raise ValueError(f"Unknown component type {t!r} in {name}")
    return MockGalaxy(name=name, redshift=record["redshift"], components=comps)


def make_config() -> ImageConfig:
    return ImageConfig(
        name="cs4g_s4g_irac1_test",
        pixel_scale=PIXEL_SCALE,
        zeropoint=ZEROPOINT,
        size_factor=SIZE_FACTOR,
        max_image_size=MAX_IMAGE_SIZE,
        psf_enabled=True,
        psf_type=PSF_TYPE,
        psf_fwhm=PSF_FWHM,
        engine="libprofit",
    )


# --------------------------------------------------------------------------- #
# S4G cube handling
# --------------------------------------------------------------------------- #

_FITSECT_RE = __import__("re").compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]")


def _parse_fitsect(obj: str) -> tuple[int, int]:
    """Parse a FITS section string like '[437:749,644:956]' and return the
    (x_lo, y_lo) 1-indexed origin. Returns (1, 1) if no match."""
    m = _FITSECT_RE.search(obj)
    if not m:
        return 1, 1
    return int(m.group(1)), int(m.group(3))


def load_s4g_reference(subcomps_path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Returns
    -------
    ref_image : 2D array, sum of model component planes (HDU2..HDU_last),
                sky already subtracted (GALFIT fits model + sky; HDU1 is
                the residual so it's not needed here).
    center_xy : (x, y) 0-indexed pixel center in cube-local coordinates.
                GALFIT stores 1_XC/1_YC in FITS-1-indexed full-mosaic
                coordinates, so we subtract the cube's FITSECT origin
                (parsed from HDU1's OBJECT keyword) to get cube-local
                coords, then -1 to shift FITS 1-indexed to Python 0-indexed.
    """
    with fits.open(subcomps_path) as hdul:
        n_hdu = len(hdul)
        comp_planes = [np.asarray(hdul[i].data, dtype=np.float64)
                       for i in range(2, n_hdu)]
        ref = np.sum(comp_planes, axis=0)
        h2 = hdul[2].header
        xc_mosaic = float(str(h2["1_XC"]).strip("[]").split()[0])
        yc_mosaic = float(str(h2["1_YC"]).strip("[]").split()[0])
        fitsect = str(hdul[1].header.get("OBJECT", "[1:1,1:1]"))
        x_lo, y_lo = _parse_fitsect(fitsect)
        xc = xc_mosaic - x_lo        # FITS 1-indexed → 0-indexed cube-local
        yc = yc_mosaic - y_lo
    return ref, (xc, yc)


def crop_around(image: np.ndarray, center_xy: tuple[float, float],
                half_size: int) -> np.ndarray:
    """Crop ``image`` to (2*half+1)^2 around the given (x, y) pixel center,
    zero-padding if the crop extends past the image edges."""
    cx, cy = center_xy
    ix, iy = int(round(cx)), int(round(cy))
    ny, nx = image.shape
    size = 2 * half_size + 1
    out = np.zeros((size, size), dtype=image.dtype)
    y0 = iy - half_size
    x0 = ix - half_size
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    dst_y1 = min(ny, y0 + size)
    dst_x1 = min(nx, x0 + size)
    hh = dst_y1 - dst_y0
    ww = dst_x1 - dst_x0
    if hh > 0 and ww > 0:
        out[src_y0:src_y0 + hh, src_x0:src_x0 + ww] = \
            image[dst_y0:dst_y0 + hh, dst_x0:dst_x0 + ww]
    return out


# --------------------------------------------------------------------------- #
# Comparison metrics
# --------------------------------------------------------------------------- #

def radial_profile(image: np.ndarray, rmax: float,
                   nbins: int = 60) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = image.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    y, x = np.mgrid[:ny, :nx]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    edges = np.linspace(0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.zeros(nbins)
    for i in range(nbins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        if mask.any():
            vals[i] = image[mask].mean()
    return centers, vals


def compute_metrics(mock: np.ndarray, ref: np.ndarray) -> dict:
    """Minimal validation metrics: total flux agreement, shape agreement,
    and a PSF-core diagnostic. Both inputs are noise-free model renders,
    same shape, centered on the galaxy, float ADU."""
    mock_sum = float(mock.sum())
    ref_sum = float(ref.sum())
    return {
        "flux_ratio": mock_sum / ref_sum if ref_sum else float("nan"),
        "corr": float(np.corrcoef(mock.ravel(), ref.ravel())[0, 1]),
        "peak_ratio": float(mock.max() / ref.max()) if ref.max() else float("nan"),
        "mock_sum": mock_sum,
        "ref_sum": ref_sum,
    }


# --------------------------------------------------------------------------- #
# Rendering + orchestration
# --------------------------------------------------------------------------- #

def render_and_compare(name: str, record: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    galaxy = build_mockgalaxy(name, record)
    config = make_config()

    color_shift = float(record.get("color_shift_3p6_minus_i", 0.0))
    print(f"[{name}] color_shift (M_3.6 - M_i) = {color_shift:+.4f} mag "
          f"(added to abs_mag before rendering)")
    print(f"[{name}] rendering via mockgal libprofit...")
    mock_img, meta = generate_mock_image(
        name=name,
        redshift=galaxy.redshift,
        components=galaxy.components,
        config=config,
    )
    mock_path = out_dir / "mockgal.fits"
    hdu = fits.PrimaryHDU(mock_img.astype(np.float32))
    hdu.header["NAME"] = name
    hdu.header["ENGINE"] = meta.get("engine", "?")
    hdu.header["PIXSCALE"] = PIXEL_SCALE
    hdu.header["ZEROPT"] = ZEROPOINT
    hdu.header["PSF_FWHM"] = PSF_FWHM
    hdu.writeto(mock_path, overwrite=True)
    print(f"[{name}] mockgal shape {mock_img.shape}, sum={mock_img.sum():.3e}")

    print(f"[{name}] loading S4G reference...")
    subcomps = out_dir / "s4g_subcomps.fits"
    ref_full, (xc, yc) = load_s4g_reference(subcomps)
    print(f"[{name}] S4G ref shape {ref_full.shape}, galaxy center "
          f"({xc:.1f}, {yc:.1f}), sum={ref_full.sum():.3e}")

    # Crop both images to the largest square that fits in both: the
    # mockgal render is sized to 6 * max_Re (often huge), the S4G cube
    # is the FITSECT Salo used during fitting. The smaller determines
    # the common FOV.
    ny_s, nx_s = ref_full.shape
    s4g_half = min(int(xc), nx_s - int(xc), int(yc), ny_s - int(yc)) - 2
    mock_half = mock_img.shape[0] // 2 - 2
    half = min(s4g_half, mock_half)
    print(f"[{name}] cropping both to half-size {half} pix "
          f"({(2*half+1)*PIXEL_SCALE:.1f}\" per side)")

    mock_cx = (mock_img.shape[1] - 1) / 2.0
    mock_cy = (mock_img.shape[0] - 1) / 2.0
    mock_crop = crop_around(mock_img, (mock_cx, mock_cy), half)
    ref_crop = crop_around(ref_full, (xc, yc), half)

    ref_path = out_dir / "s4g_reference.fits"
    rhdu = fits.PrimaryHDU(ref_crop.astype(np.float32))
    rhdu.header["NAME"] = name
    rhdu.header["SOURCE"] = "Salo+2015 S4G subcomps sum"
    rhdu.writeto(ref_path, overwrite=True)

    max_re_kpc = max(
        (c.get("r_eff_kpc") or c.get("r_out_kpc") or 0.0)
        for c in record["components"]
    )
    metrics = compute_metrics(mock_crop, ref_crop)
    metrics["half_size_pix"] = int(half)
    metrics["max_re_kpc"] = float(max_re_kpc)
    print(f"[{name}] metrics: {metrics}")

    return {
        "name": name,
        "components": [c["type"] for c in record["components"]],
        "color_shift_3p6_minus_i": color_shift,
        "mock_path": str(mock_path),
        "ref_path": str(ref_path),
        "subcomps_path": str(subcomps),
        "mock_shape": list(mock_img.shape),
        "ref_shape": list(ref_full.shape),
        "center_s4g": [xc, yc],
        "metrics": metrics,
        "mock_crop": mock_crop,
        "ref_crop": ref_crop,
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def make_qa_figure(results: list[dict], out_path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i, r in enumerate(results):
        a = r["mock_crop"]
        b = r["ref_crop"]
        diff = a - b
        pos_vals = a[a > 0]
        vmin = max(np.percentile(pos_vals, 1), 1e-6) if pos_vals.size else 1e-6
        vmax = max(a.max(), b.max())

        axes[i, 0].imshow(a, origin="lower", norm=LogNorm(vmin=vmin, vmax=vmax),
                          cmap="magma")
        axes[i, 0].set_title(f"{r['name']} — mockgal libprofit "
                             f"({'+'.join(r['components'])})")

        axes[i, 1].imshow(b, origin="lower", norm=LogNorm(vmin=vmin, vmax=vmax),
                          cmap="magma")
        axes[i, 1].set_title(f"{r['name']} — S4G reference")

        vlim = np.percentile(np.abs(diff), 99)
        axes[i, 2].imshow(diff, origin="lower", cmap="seismic",
                          vmin=-vlim, vmax=+vlim)
        m = r["metrics"]
        axes[i, 2].set_title(
            f"mock − s4g  (corr={m.get('corr', float('nan')):.3f}, "
            f"flux ratio {m.get('flux_ratio', float('nan')):.3f})"
        )

        ny = a.shape[0]
        rmax = ny // 2
        r_grid, prof_m = radial_profile(a, rmax)
        _, prof_s = radial_profile(b, rmax)
        axes[i, 3].semilogy(r_grid * PIXEL_SCALE,
                            np.clip(prof_m, 1e-6, None),
                            label="mockgal", color="tab:blue")
        axes[i, 3].semilogy(r_grid * PIXEL_SCALE,
                            np.clip(prof_s, 1e-6, None),
                            label="S4G ref", color="tab:orange",
                            linestyle="--")
        axes[i, 3].set_xlabel("radius [arcsec]")
        axes[i, 3].set_ylabel("mean intensity")
        axes[i, 3].set_title(f"{r['name']} — radial profile")
        axes[i, 3].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"QA figure saved: {out_path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--galaxies", nargs="+", default=TARGETS,
                   help=f"Galaxy names to validate (default: {TARGETS})")
    p.add_argument("--output-root", type=Path, default=OUT_ROOT)
    p.add_argument("--model-yaml", type=Path, default=MODEL_YAML)
    args = p.parse_args()

    model_dict = load_model_dict(args.model_yaml)

    results = []
    for name in args.galaxies:
        if name not in model_dict:
            print(f"[{name}] SKIP: not in {args.model_yaml}")
            continue
        out_dir = args.output_root / name
        if not (out_dir / "s4g_subcomps.fits").exists():
            print(f"[{name}] SKIP: {out_dir / 's4g_subcomps.fits'} missing; "
                  "fetch the S4G cube first")
            continue
        results.append(render_and_compare(name, model_dict[name], out_dir))

    if not results:
        print("No galaxies rendered.")
        return 1

    qa_path = args.output_root / "qa_s4g_validation.png"
    make_qa_figure(results, qa_path)

    summary = {
        r["name"]: {k: v for k, v in r.items()
                    if k not in ("mock_crop", "ref_crop")}
        for r in results
    }
    summary_path = args.output_root / "qa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
