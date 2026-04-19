#!/usr/bin/env python
"""
qa_libprofit_vs_galfit.py - Side-by-side QA for CS4G mock renders.

For each requested galaxy, loads the mockgal MockGalaxy from
`inputs/cs4g/cs4g_components.json`, renders the same MockGalaxy via
both the mockgal libprofit path and the mockgal_galfit wrapper, saves
the two FITS images to `output/cs4g_qa/{name}/`, and emits a PNG
comparison figure.

Usage:
    python inputs/cs4g/scripts/qa_libprofit_vs_galfit.py \
        --galaxies NGC1097 NGC1365
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Make sure libprofit's dylib can be found (this profit-cli build has a
# hardcoded @rpath that resolves to a non-existent /Users/mac/... path).
LIBPROFIT_DIR = REPO_ROOT / "libprofit" / "build"
os.environ.setdefault("LIBPROFIT_PATH", str(LIBPROFIT_DIR))
_extra_dylib = str(LIBPROFIT_DIR)
existing = os.environ.get("DYLD_LIBRARY_PATH", "")
if _extra_dylib not in existing.split(":"):
    os.environ["DYLD_LIBRARY_PATH"] = (
        f"{_extra_dylib}:{existing}" if existing else _extra_dylib
    )

from mockgal import (
    FerrerComponent,
    ImageConfig,
    MockGalaxy,
    PointSourceComponent,
    SersicComponent,
    generate_mock_image,
)
from mockgal_galfit import render_with_galfit


DEFAULT_COMPONENTS = REPO_ROOT / "inputs" / "cs4g" / "cs4g_components.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "cs4g_qa"

# Render settings. S4G IRAC1 native pixel scale is 0.75"/pix, but we
# down-sample to an HSC-like 0.168"/pix to match the main Huang2013
# workflow. Images stay large enough to contain the outermost disk.
PIXEL_SCALE = 0.75
ZEROPOINT = 21.097   # S4G IRAC1 zeropoint (matches GALFIT K))
PSF_FWHM = 1.8       # approximate S4G PSF FWHM (arcsec)
SIZE_PIXELS = 401    # plenty of room; all CS4G disks fit inside ~300 pix


def build_mockgalaxy(name: str, record: dict) -> MockGalaxy:
    """Build a MockGalaxy from a cs4g_components.json entry."""
    comps = []
    for c in record["components"]:
        t = c["type"]
        cid = c.get("id")
        if t == "sersic":
            comps.append(SersicComponent(
                r_eff_kpc=c["r_eff_kpc"],
                abs_mag=c["abs_mag"],
                n=c["n"],
                ellipticity=c["ellipticity"],
                pa_deg=c["pa_deg"],
                component_id=cid,
            ))
        elif t == "ferrer":
            comps.append(FerrerComponent(
                r_out_kpc=c["r_out_kpc"],
                abs_mag=c["abs_mag"],
                alpha=c["alpha"],
                beta=c["beta"],
                ellipticity=c["ellipticity"],
                pa_deg=c["pa_deg"],
                component_id=cid,
            ))
        elif t == "psf":
            comps.append(PointSourceComponent(
                abs_mag=c["abs_mag"],
                component_id=cid,
            ))
        else:
            raise ValueError(f"Unknown component type {t!r} for {name}")
    return MockGalaxy(name=name, redshift=record["redshift"], components=comps)


def make_config() -> ImageConfig:
    return ImageConfig(
        name="cs4g_qa",
        pixel_scale=PIXEL_SCALE,
        zeropoint=ZEROPOINT,
        size_pixels=SIZE_PIXELS,
        psf_enabled=True,
        psf_type="gaussian",
        psf_fwhm=PSF_FWHM,
    )


def radial_profile(image: np.ndarray, center: tuple[float, float],
                   rmax: float, nbins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Mean pixel value in annular bins out to rmax (pixels)."""
    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    edges = np.linspace(0, rmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.zeros(nbins)
    for i in range(nbins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        if mask.any():
            vals[i] = image[mask].mean()
    return centers, vals


def render_one(name: str, components_record: dict, out_dir: Path) -> dict:
    """Render `name` via both backends and save the FITS outputs. Return a dict
    of metadata + filepaths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    galaxy = build_mockgalaxy(name, components_record)
    config = make_config()

    print(f"[{name}] rendering via mockgal libprofit...")
    img_lp, meta_lp = generate_mock_image(
        name=name,
        redshift=galaxy.redshift,
        components=galaxy.components,
        config=config,
    )
    lp_path = out_dir / "libprofit.fits"
    hdu = fits.PrimaryHDU(img_lp.astype(np.float32))
    hdu.header["ENGINE"] = (meta_lp.get("engine", "?"), "render backend")
    hdu.header["NAME"] = name
    hdu.writeto(lp_path, overwrite=True)

    print(f"[{name}] rendering via mockgal_galfit (GALFIT binary)...")
    img_gf, meta_gf = render_with_galfit(galaxy, config)
    gf_path = out_dir / "galfit.fits"
    hdu = fits.PrimaryHDU(img_gf.astype(np.float32))
    hdu.header["ENGINE"] = ("galfit", "render backend")
    hdu.header["NAME"] = name
    hdu.writeto(gf_path, overwrite=True)

    stats = compute_stats(img_lp, img_gf)
    print(f"[{name}] correlation={stats['corr']:.5f}, "
          f"flux_ratio={stats['flux_ratio']:.5f}, "
          f"peak_ratio={stats['peak_ratio']:.5f}")
    return {
        "name": name,
        "libprofit": str(lp_path),
        "galfit": str(gf_path),
        "libprofit_engine": meta_lp.get("engine"),
        "stats": stats,
        "image_lp": img_lp,
        "image_gf": img_gf,
    }


def compute_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a_flat = a.flatten()
    b_flat = b.flatten()
    return {
        "shape": a.shape,
        "sum_lp": float(a.sum()),
        "sum_gf": float(b.sum()),
        "max_lp": float(a.max()),
        "max_gf": float(b.max()),
        "flux_ratio": float(a.sum() / b.sum()) if b.sum() else float("nan"),
        "peak_ratio": float(a.max() / b.max()) if b.max() else float("nan"),
        "corr": float(np.corrcoef(a_flat, b_flat)[0, 1]),
        "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
    }


def make_qa_figure(results: list[dict], out_path: Path) -> None:
    """3-column per galaxy: libprofit, galfit, (lp - gf) / max; plus radial."""
    n_gal = len(results)
    fig, axes = plt.subplots(n_gal, 4, figsize=(16, 4 * n_gal))
    if n_gal == 1:
        axes = axes[None, :]

    for row_i, res in enumerate(results):
        a = res["image_lp"]
        b = res["image_gf"]
        diff = a - b
        vmin = max(np.percentile(a[a > 0], 1), 1e-6) if (a > 0).any() else 1e-6
        vmax = max(a.max(), b.max())

        axes[row_i, 0].imshow(a, origin="lower", norm=LogNorm(vmin=vmin, vmax=vmax),
                              cmap="magma")
        axes[row_i, 0].set_title(f"{res['name']} — mockgal libprofit")

        axes[row_i, 1].imshow(b, origin="lower", norm=LogNorm(vmin=vmin, vmax=vmax),
                              cmap="magma")
        axes[row_i, 1].set_title(f"{res['name']} — mockgal_galfit")

        vlim = np.percentile(np.abs(diff), 99)
        axes[row_i, 2].imshow(diff, origin="lower", cmap="seismic",
                              vmin=-vlim, vmax=+vlim)
        axes[row_i, 2].set_title(
            f"libprofit − galfit  (corr={res['stats']['corr']:.4f}, "
            f"flux ratio {res['stats']['flux_ratio']:.3f})"
        )

        ny, nx = a.shape
        rmax = min(ny, nx) // 2
        r, prof_a = radial_profile(a, (ny / 2, nx / 2), rmax)
        _, prof_b = radial_profile(b, (ny / 2, nx / 2), rmax)
        axes[row_i, 3].semilogy(r, np.clip(prof_a, 1e-6, None),
                                label="libprofit", color="tab:blue")
        axes[row_i, 3].semilogy(r, np.clip(prof_b, 1e-6, None),
                                label="galfit", color="tab:orange",
                                linestyle="--")
        axes[row_i, 3].set_xlabel("radius [pix]")
        axes[row_i, 3].set_ylabel("mean intensity (log)")
        axes[row_i, 3].set_title(f"{res['name']} — radial profile")
        axes[row_i, 3].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"QA figure saved: {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--galaxies", type=str, nargs="+", required=True,
                   help="Galaxy names from cs4g_components.json")
    args = p.parse_args()

    data = json.loads(args.components.read_text())
    results = []
    for name in args.galaxies:
        if name not in data:
            print(f"[{name}] SKIP: not in components JSON")
            continue
        out_dir = args.output_dir / name
        results.append(render_one(name, data[name], out_dir))

    if not results:
        print("No galaxies rendered.")
        return 1

    qa_path = args.output_dir / "qa_libprofit_vs_galfit.png"
    make_qa_figure(results, qa_path)

    # Summary JSON for future references.
    summary = {
        r["name"]: {k: v for k, v in r.items() if k not in ("image_lp", "image_gf")}
        for r in results
    }
    summary_path = args.output_dir / "qa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
