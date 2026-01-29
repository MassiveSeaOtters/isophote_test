#!/usr/bin/env python
"""
mockgal.py - Mock Galaxy Image Generator

Generate mock galaxy images with multi-component Sersic profiles for testing
image analysis algorithms. Supports PSF convolution, sky background, and noise.

Input files:
    - Model file: YAML/JSON containing galaxy definitions (name, redshift, components)
    - Config file: YAML/JSON containing image generation settings (PSF, sky, noise)

Usage:
    # Single galaxy mode (CLI parameters)
    python mockgal.py --single -z 0.01 --r-eff 1.0 --abs-mag -20 --sersic-n 4

    # Using model file with default settings
    python mockgal.py --models galaxies.yaml

    # Using model file with custom config
    python mockgal.py --models galaxies.yaml --config image_config.yaml

    # Select specific galaxies from model file
    python mockgal.py --models galaxies.yaml --galaxy NGC1399 IC1459
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma, gammaincinv

# Astropy imports
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.modeling.models import Sersic2D

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Optional matplotlib support
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from scipy.ndimage import gaussian_filter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# =============================================================================
# Section 1: Constants and Logging
# =============================================================================

DEFAULT_PIXEL_SCALE = 0.3  # arcsec/pixel
DEFAULT_ZEROPOINT = 27.0   # mag
DEFAULT_SIZE_FACTOR = 15.0 # image half-size = factor * max(Re)
DEFAULT_SIZE_PIXELS = 2001 # default fixed size for large galaxies
DEFAULT_PSF_FWHM = 1.0     # arcsec
DEFAULT_MOFFAT_BETA = 4.765
DEFAULT_REDSHIFT = 0.01
MAX_SERSIC_INDEX = 8.0
MAX_IMAGE_SIZE = 4001      # maximum image dimension to avoid memory issues

# Default cosmology
DEFAULT_H0 = 70.0  # km/s/Mpc
DEFAULT_OM = 0.3

# profit-cli path (can be overridden via CLI or environment variable)
LIBPROFIT_PATH = os.environ.get('LIBPROFIT_PATH', None)
PROFIT_CLI_PATH = os.environ.get('PROFIT_CLI_PATH', None)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Section 2: Data Classes
# =============================================================================

@dataclass
class SersicComponent:
    """Single Sersic component with intrinsic parameters."""
    r_eff_kpc: float      # Effective radius in kpc
    abs_mag: float        # Absolute magnitude
    n: float              # Sersic index
    ellipticity: float = 0.0  # 1 - b/a
    pa_deg: float = 0.0   # Position angle in degrees
    component_id: Optional[str] = None  # Optional component identifier
    index: Optional[int] = None         # Optional component index

    def __post_init__(self):
        if self.r_eff_kpc <= 0:
            raise ValueError(f"r_eff_kpc must be positive, got {self.r_eff_kpc}")
        if self.n <= 0:
            raise ValueError(f"Sersic index n must be positive, got {self.n}")
        if self.n > MAX_SERSIC_INDEX:
            raise ValueError(f"Sersic index n must be <= {MAX_SERSIC_INDEX}, got {self.n}")
        if not (0 <= self.ellipticity < 1):
            raise ValueError(f"Ellipticity must be in [0, 1), got {self.ellipticity}")

    @property
    def axrat(self) -> float:
        """Axis ratio b/a."""
        return 1.0 - self.ellipticity


@dataclass
class MockGalaxy:
    """Complete mock galaxy definition."""
    name: str
    redshift: float
    components: List[SersicComponent]

    def __post_init__(self):
        if self.redshift <= 0:
            raise ValueError(f"Redshift must be positive, got {self.redshift}")
        if not self.components:
            raise ValueError("Galaxy must have at least one component")

    @property
    def total_abs_mag(self) -> float:
        """Compute total absolute magnitude from components."""
        fluxes = [10 ** (-0.4 * c.abs_mag) for c in self.components]
        return -2.5 * np.log10(sum(fluxes))


@dataclass
class ImageConfig:
    """Configuration for image generation."""
    name: str = "default"
    pixel_scale: float = DEFAULT_PIXEL_SCALE
    zeropoint: float = DEFAULT_ZEROPOINT
    size_factor: float = DEFAULT_SIZE_FACTOR
    size_pixels: Optional[int] = None

    # PSF configuration
    psf_enabled: bool = False
    psf_type: str = "gaussian"  # gaussian | moffat | image
    psf_fwhm: float = DEFAULT_PSF_FWHM
    psf_moffat_beta: float = DEFAULT_MOFFAT_BETA
    psf_file: Optional[str] = None

    # Sky configuration
    sky_enabled: bool = False
    sky_type: str = "flat"  # flat | tilted
    sky_level: float = 0.0
    sky_coeffs: List[float] = field(default_factory=lambda: [0.0])

    # Noise configuration
    noise_enabled: bool = False
    noise_sigma: Optional[float] = None
    noise_snr: Optional[float] = None
    noise_seed: Optional[int] = None

    # Engine selection
    engine: str = "auto"  # libprofit | astropy | auto
    profit_cli_path: Optional[str] = None  # Custom path to profit-cli

    def __post_init__(self):
        if self.pixel_scale <= 0:
            raise ValueError(f"pixel_scale must be positive, got {self.pixel_scale}")
        if self.psf_type not in ("gaussian", "moffat", "image"):
            raise ValueError(f"psf_type must be gaussian|moffat|image, got {self.psf_type}")
        if self.sky_type not in ("flat", "tilted"):
            raise ValueError(f"sky_type must be flat|tilted, got {self.sky_type}")
        if self.engine not in ("libprofit", "astropy", "auto"):
            raise ValueError(f"engine must be libprofit|astropy|auto, got {self.engine}")


# =============================================================================
# Section 3: Cosmology Utilities
# =============================================================================

def get_cosmology(H0: float = DEFAULT_H0, Om: float = DEFAULT_OM) -> FlatLambdaCDM:
    """Get astropy cosmology object."""
    return FlatLambdaCDM(H0=H0, Om0=Om)


def angular_diameter_distance(z: float, H0: float = DEFAULT_H0, Om: float = DEFAULT_OM) -> float:
    """
    Compute angular diameter distance in Mpc.

    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float
        Matter density parameter

    Returns
    -------
    float
        Angular diameter distance in Mpc
    """
    cosmo = get_cosmology(H0, Om)
    return cosmo.angular_diameter_distance(z).value


def kpc_to_arcsec(r_kpc: float, z: float) -> float:
    """
    Convert physical size (kpc) to angular size (arcsec).

    Parameters
    ----------
    r_kpc : float
        Physical size in kpc
    z : float
        Redshift

    Returns
    -------
    float
        Angular size in arcsec
    """
    d_a = angular_diameter_distance(z)  # Mpc
    # r_kpc / (d_a * 1000) gives radians, convert to arcsec
    return r_kpc / (d_a * 1000) * 206264.806


def abs_to_app_mag(abs_mag: float, z: float, k_corr: float = 0.0) -> float:
    """
    Convert absolute magnitude to apparent magnitude.

    Parameters
    ----------
    abs_mag : float
        Absolute magnitude
    z : float
        Redshift
    k_corr : float
        K-correction (default 0)

    Returns
    -------
    float
        Apparent magnitude
    """
    cosmo = get_cosmology()
    dist_mod = cosmo.distmod(z).value
    return abs_mag + dist_mod + k_corr


# =============================================================================
# Section 4: Sersic Engine
# =============================================================================

def _resolve_profit_cli_path(path_value: Optional[str]) -> Optional[str]:
    """Resolve a profit-cli path that may be a directory or a full path."""
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_dir():
        path = path / "profit-cli"
    if path.is_file():
        if os.access(path, os.X_OK):
            return str(path)
        logger.warning(f"profit-cli found but not executable: {path}")
    return None


def find_profit_cli(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Find the profit-cli executable.

    Parameters
    ----------
    custom_path : str, optional
        Custom path to profit-cli binary

    Returns
    -------
    str or None
        Path to profit-cli if found, None otherwise
    """
    # Check custom path first
    resolved = _resolve_profit_cli_path(custom_path)
    if resolved:
        return resolved

    # Check environment variable
    resolved = _resolve_profit_cli_path(LIBPROFIT_PATH)
    if resolved:
        return resolved
    resolved = _resolve_profit_cli_path(PROFIT_CLI_PATH)
    if resolved:
        return resolved

    # Check PATH
    profit_in_path = shutil.which("profit-cli")
    if profit_in_path:
        return profit_in_path

    # Check common locations
    common_locations = [
        Path(__file__).parent / "libprofit" / "build" / "profit-cli",
        Path.home() / "libprofit" / "build" / "profit-cli",
        Path("/usr/local/bin/profit-cli"),
        Path("/opt/homebrew/bin/profit-cli"),
    ]
    for loc in common_locations:
        if loc.is_file():
            return str(loc)

    return None


class SersicEngine:
    """
    Sersic profile rendering engine.

    Supports libprofit (via profit-cli), and astropy (fallback) backends.
    """

    def __init__(self, engine: str = "auto", profit_cli_path: Optional[str] = None):
        """
        Initialize the Sersic engine.

        Parameters
        ----------
        engine : str
            Engine to use: 'libprofit', 'astropy', or 'auto'
        profit_cli_path : str, optional
            Custom path to profit-cli binary
        """
        self.profit_cli_path = find_profit_cli(profit_cli_path)
        self.engine = self._select_engine(engine)
        logger.info(f"Using Sersic engine: {self.engine}")

    def _select_engine(self, requested: str) -> str:
        """Select the best available engine."""
        if requested == "auto":
            if self.profit_cli_path:
                return "libprofit"
            else:
                logger.warning("profit-cli not found, using astropy")
                return "astropy"
        elif requested == "libprofit":
            if self.profit_cli_path:
                return "libprofit"
            logger.warning(
                "libprofit requested but profit-cli not found. "
                "Falling back to astropy."
            )
            return "astropy"
        elif requested == "astropy":
            return "astropy"
        else:
            raise ValueError(f"Unknown engine: {requested}")

    def render(
        self,
        shape: Tuple[int, int],
        xcen: float,
        ycen: float,
        mag: float,
        re_pix: float,
        n: float,
        axrat: float,
        ang: float,
        zeropoint: float,
        psf: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render a single Sersic component.

        Parameters
        ----------
        shape : tuple
            Image shape (ny, nx)
        xcen, ycen : float
            Center coordinates in pixels
        mag : float
            Total apparent magnitude
        re_pix : float
            Effective radius in pixels
        n : float
            Sersic index
        axrat : float
            Axis ratio (b/a)
        ang : float
            Position angle in degrees (from +Y, CCW)
        zeropoint : float
            Photometric zeropoint
        psf : ndarray, optional
            PSF kernel for convolution

        Returns
        -------
        ndarray
            Rendered 2D image
        """
        if self.engine == "libprofit":
            return self._render_libprofit(
                shape, xcen, ycen, mag, re_pix, n, axrat, ang, zeropoint, psf
            )
        else:
            return self._render_astropy(
                shape, xcen, ycen, mag, re_pix, n, axrat, ang, zeropoint, psf
            )

    def _render_libprofit(
        self,
        shape: Tuple[int, int],
        xcen: float,
        ycen: float,
        mag: float,
        re_pix: float,
        n: float,
        axrat: float,
        ang: float,
        zeropoint: float,
        psf: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render using libprofit via profit-cli.

        profit-cli parameters for Sersic:
        -p sersic:xcen=X:ycen=Y:mag=M:re=R:nser=N:axrat=A:ang=PA
        -w WIDTH -H HEIGHT -m ZEROPOINT
        """
        height, width = shape

        # Build profile specification
        profile_spec = (
            f"sersic:xcen={xcen}:ycen={ycen}:mag={mag}:"
            f"re={re_pix}:nser={n}:axrat={axrat}:ang={ang}"
        )

        # Build command
        cmd = [
            self.profit_cli_path,
            "-w", str(width),   # Width in pixels
            "-H", str(height),  # Height in pixels
            "-m", str(zeropoint),
            "-p", profile_spec,
            "-t"  # Output as text to stdout
        ]

        # Handle PSF if provided
        psf_file = None
        if psf is not None:
            # Write PSF to temporary FITS file
            psf_file = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
            psf_hdu = fits.PrimaryHDU(psf.astype(np.float32))
            psf_hdu.writeto(psf_file.name, overwrite=True)
            psf_file.close()
            cmd.extend(["-P", psf_file.name])

        try:
            # Run profit-cli
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output - profit-cli outputs timing info in first 2 lines,
            # then space-separated pixel values, one row per line
            lines = result.stdout.strip().split('\n')

            # Skip header lines (timing info starts with "Created" or "Ran")
            data_lines = []
            for line in lines:
                if line.startswith(('Created', 'Ran')):
                    continue
                # Try to parse as numeric data
                try:
                    vals = [float(v) for v in line.split()]
                    if vals:  # Non-empty row
                        data_lines.append(vals)
                except ValueError:
                    continue

            if not data_lines:
                raise RuntimeError("No valid image data in profit-cli output")

            image = np.array(data_lines, dtype=np.float64)

            # Verify shape matches expected
            if image.shape != (height, width):
                logger.warning(
                    f"profit-cli output shape {image.shape} differs from expected {(height, width)}"
                )

            return image

        except subprocess.CalledProcessError as e:
            logger.error(f"profit-cli failed: {e.stderr}")
            raise RuntimeError(f"profit-cli execution failed: {e.stderr}")

        finally:
            # Clean up temporary PSF file
            if psf_file is not None:
                try:
                    os.unlink(psf_file.name)
                except OSError:
                    pass

    def _render_astropy(
        self,
        shape: Tuple[int, int],
        xcen: float,
        ycen: float,
        mag: float,
        re_pix: float,
        n: float,
        axrat: float,
        ang: float,
        zeropoint: float,
        psf: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render using astropy.modeling.models.Sersic2D.

        Convention differences from pyprofit:
        - ellip = 1 - axrat (astropy uses ellipticity)
        - theta: angle from +X axis in radians (pyprofit: from +Y in degrees)
        - amplitude: intensity at Re (pyprofit uses total magnitude)
        """
        # Convert magnitude to amplitude (intensity at Re)
        # Total flux: F_tot = 2 * pi * n * I_e * R_e^2 * exp(b_n) * gamma(2n) / b_n^(2n)
        b_n = gammaincinv(2 * n, 0.5)
        total_flux = 10 ** (-0.4 * (mag - zeropoint))
        gamma_2n = gamma(2 * n)

        # Solve for amplitude (I_e at the effective radius)
        # Include axrat factor for elliptical profile
        amplitude = total_flux * b_n ** (2 * n) / (
            2 * np.pi * n * re_pix**2 * np.exp(b_n) * gamma_2n * axrat
        )

        # Convention conversion
        ellip = 1 - axrat
        # pyprofit: angle from +Y CCW in degrees
        # astropy: angle from +X CCW in radians
        theta = np.radians(90 - ang)

        model = Sersic2D(
            amplitude=amplitude,
            r_eff=re_pix,
            n=n,
            x_0=xcen,
            y_0=ycen,
            ellip=ellip,
            theta=theta
        )

        y, x = np.mgrid[:shape[0], :shape[1]]
        image = model(x, y)

        # Handle PSF convolution separately for astropy
        if psf is not None:
            image = convolve(image, psf, mode='constant')

        return image


# =============================================================================
# Section 5: Image Generator
# =============================================================================

class MockImageGenerator:
    """Main class for generating mock galaxy images."""

    def __init__(self, config: ImageConfig):
        """
        Initialize the image generator.

        Parameters
        ----------
        config : ImageConfig
            Image generation configuration
        """
        self.config = config
        self.engine = SersicEngine(config.engine, config.profit_cli_path)

    def generate(self, galaxy: MockGalaxy) -> Tuple[np.ndarray, dict]:
        """
        Generate a mock image for a galaxy.

        Parameters
        ----------
        galaxy : MockGalaxy
            Galaxy definition

        Returns
        -------
        image : ndarray
            2D image array
        metadata : dict
            Image metadata including WCS info and component parameters
        """
        # 1. Compute derived parameters
        params = self._compute_derived_params(galaxy)

        # 2. Determine image size
        shape = self._compute_image_shape(params)

        # 3. Prepare PSF if needed
        psf = self._make_psf() if self.config.psf_enabled else None

        # 4. Render all components
        image = np.zeros(shape, dtype=np.float64)
        xcen = shape[1] / 2.0
        ycen = shape[0] / 2.0

        for comp, comp_params in zip(galaxy.components, params['components']):
            # For pyprofit, pass PSF for native convolution
            # For astropy, we'll convolve the full image at the end
            psf_for_render = psf if self.engine.engine == "libprofit" else None

            image += self.engine.render(
                shape=shape,
                xcen=xcen,
                ycen=ycen,
                mag=comp_params['app_mag'],
                re_pix=comp_params['re_pix'],
                n=comp.n,
                axrat=comp.axrat,
                ang=comp.pa_deg,
                zeropoint=self.config.zeropoint,
                psf=psf_for_render
            )

        # 5. Apply PSF for astropy engine (done per-component for pyprofit)
        if self.config.psf_enabled and self.engine.engine == "astropy":
            image = convolve(image, psf, mode='constant')

        # 6. Add sky background
        if self.config.sky_enabled:
            image += self._make_sky(shape)

        # 7. Add noise
        if self.config.noise_enabled:
            image = self._add_noise(image, params)

        # 8. Build metadata
        metadata = self._build_metadata(galaxy, params, shape)

        return image, metadata

    def _compute_derived_params(self, galaxy: MockGalaxy) -> dict:
        """Convert intrinsic to observable parameters."""
        params = {'components': []}

        max_re_arcsec = 0
        for comp in galaxy.components:
            re_arcsec = kpc_to_arcsec(comp.r_eff_kpc, galaxy.redshift)
            re_pix = re_arcsec / self.config.pixel_scale
            app_mag = abs_to_app_mag(comp.abs_mag, galaxy.redshift)

            params['components'].append({
                're_arcsec': re_arcsec,
                're_pix': re_pix,
                'app_mag': app_mag,
                'r_eff_kpc': comp.r_eff_kpc,
                'abs_mag': comp.abs_mag,
                'n': comp.n,
                'ellipticity': comp.ellipticity,
                'pa_deg': comp.pa_deg
            })
            max_re_arcsec = max(max_re_arcsec, re_arcsec)

        params['max_re_arcsec'] = max_re_arcsec
        params['max_re_pix'] = max_re_arcsec / self.config.pixel_scale
        params['redshift'] = galaxy.redshift

        return params

    def _compute_image_shape(self, params: dict) -> Tuple[int, int]:
        """Determine image dimensions."""
        if self.config.size_pixels is not None:
            size = self.config.size_pixels
        else:
            # size_factor times the largest Re
            half_size = int(self.config.size_factor * params['max_re_pix'])
            size = 2 * half_size + 1  # Odd for centering

        # Check for excessively large images and apply cap
        if size > MAX_IMAGE_SIZE:
            logger.warning(
                f"Computed image size {size}x{size} exceeds maximum ({MAX_IMAGE_SIZE}). "
                f"Capping to {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE} pixels."
            )
            size = MAX_IMAGE_SIZE

        return (size, size)

    def _make_psf(self) -> np.ndarray:
        """Generate or load PSF image."""
        if self.config.psf_type == "image":
            if self.config.psf_file is None:
                raise ValueError("psf_file required when psf_type='image'")
            return fits.getdata(self.config.psf_file).astype(np.float64)

        fwhm_pix = self.config.psf_fwhm / self.config.pixel_scale
        size = int(10 * fwhm_pix) | 1  # Odd size, at least 10x FWHM
        size = max(size, 11)  # Minimum size

        y, x = np.mgrid[:size, :size] - size // 2
        r2 = x**2 + y**2

        if self.config.psf_type == "gaussian":
            sigma = fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
            psf = np.exp(-r2 / (2 * sigma**2))
        else:  # moffat
            beta = self.config.psf_moffat_beta
            alpha = fwhm_pix / (2 * np.sqrt(2**(1/beta) - 1))
            psf = (1 + r2 / alpha**2) ** (-beta)

        return psf / psf.sum()  # Normalize to unity

    def _make_sky(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate sky background."""
        if self.config.sky_type == "flat":
            return np.full(shape, self.config.sky_level, dtype=np.float64)

        # 2nd order polynomial: a + bx + cy + dxy + ex^2 + fy^2
        y, x = np.mgrid[:shape[0], :shape[1]]
        x = x - shape[1] / 2.0
        y = y - shape[0] / 2.0

        # Pad coefficients to 6 elements
        c = list(self.config.sky_coeffs) + [0.0] * 6
        sky = c[0] + c[1]*x + c[2]*y + c[3]*x*y + c[4]*x**2 + c[5]*y**2
        return sky.astype(np.float64)

    def _add_noise(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Add Gaussian noise to the image."""
        rng = np.random.default_rng(self.config.noise_seed)

        if self.config.noise_sigma is not None:
            sigma = self.config.noise_sigma
        elif self.config.noise_snr is not None:
            # Compute sigma from target S/N at Re of largest component
            max_re_pix = params['max_re_pix']
            center_y = image.shape[0] // 2
            center_x = image.shape[1] // 2
            # Sample intensity at Re along +X
            x_at_re = min(center_x + int(max_re_pix), image.shape[1] - 1)
            i_re = image[center_y, x_at_re]
            sigma = max(i_re / self.config.noise_snr, 1e-10)
        else:
            raise ValueError("Either noise_sigma or noise_snr must be specified")

        return image + rng.normal(0, sigma, image.shape)

    def _build_metadata(
        self,
        galaxy: MockGalaxy,
        params: dict,
        shape: Tuple[int, int]
    ) -> dict:
        """Build metadata dictionary."""
        return {
            'name': galaxy.name,
            'redshift': galaxy.redshift,
            'pixel_scale': self.config.pixel_scale,
            'zeropoint': self.config.zeropoint,
            'image_size': shape,
            'engine': self.engine.engine,
            'psf_enabled': self.config.psf_enabled,
            'psf_type': self.config.psf_type if self.config.psf_enabled else None,
            'psf_fwhm': self.config.psf_fwhm if self.config.psf_enabled else None,
            'sky_enabled': self.config.sky_enabled,
            'noise_enabled': self.config.noise_enabled,
            'components': params['components'],
            'config_name': self.config.name
        }


# =============================================================================
# Section 6: Model File Loader
# =============================================================================

def load_file(filepath: str) -> dict:
    """Load YAML or JSON file."""
    path = Path(filepath)

    if path.suffix.lower() in ('.yaml', '.yml'):
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML files. Install with: pip install pyyaml")
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def load_model_file(
    filepath: str,
    galaxy_names: Optional[List[str]] = None
) -> List[MockGalaxy]:
    """
    Load galaxies from an Input Model File (YAML/JSON).

    The model file format:
    ```yaml
    galaxies:
      - name: "NGC1399"
        redshift: 0.001
        components:
          - id: "NGC1399_comp0"
            index: 0
            r_eff_kpc: 0.26
            abs_mag: -17.32
            n: 0.59
            ellipticity: 0.05
            pa_deg: 70.0
    ```

    Parameters
    ----------
    filepath : str
        Path to model file (.yaml or .json)
    galaxy_names : list, optional
        If provided, only load these galaxies

    Returns
    -------
    list
        List of MockGalaxy objects
    """
    data = load_file(filepath)

    # Handle both formats: with 'galaxies' key or direct list
    if isinstance(data, dict):
        galaxy_list = data.get('galaxies', [])
    else:
        galaxy_list = data

    galaxies = []
    for gal_dict in galaxy_list:
        name = gal_dict.get('name')

        # Filter by name if specified
        if galaxy_names is not None and name not in galaxy_names:
            continue

        redshift = gal_dict.get('redshift', DEFAULT_REDSHIFT)

        components = []
        for i, comp_dict in enumerate(gal_dict.get('components', [])):
            # Clamp Sersic index if needed
            n = comp_dict.get('n', 4.0)
            if n > MAX_SERSIC_INDEX:
                logger.warning(f"Galaxy {name}: Clamping n={n} to {MAX_SERSIC_INDEX}")
                n = MAX_SERSIC_INDEX

            components.append(SersicComponent(
                r_eff_kpc=comp_dict['r_eff_kpc'],
                abs_mag=comp_dict['abs_mag'],
                n=n,
                ellipticity=comp_dict.get('ellipticity', 0.0),
                pa_deg=comp_dict.get('pa_deg', 0.0),
                component_id=comp_dict.get('id'),
                index=comp_dict.get('index', i)
            ))

        if components:
            galaxies.append(MockGalaxy(
                name=name,
                redshift=redshift,
                components=components
            ))

    logger.info(f"Loaded {len(galaxies)} galaxies from {filepath}")
    return galaxies


def parse_huang2013(filepath: str) -> Dict[str, List[SersicComponent]]:
    """
    Parse the Huang 2013 ASCII catalog and return SersicComponent lists.

    This is a utility function primarily for testing. For production use,
    prefer using the convert_huang2013.py script to generate YAML files
    and then load them with load_model_file().

    Parameters
    ----------
    filepath : str
        Path to huang2013_cgs_model.txt

    Returns
    -------
    dict
        Dictionary mapping galaxy names to lists of SersicComponent objects

    Notes
    -----
    The Huang 2013 catalog contains galaxies with multiple Sersic components.
    The VMag column is already in absolute magnitudes, not apparent.
    Default redshift is 0.01 for all galaxies.
    """
    HUANG_REDSHIFT = 0.01
    galaxies = {}

    with open(filepath, 'r') as f:
        for line in f:
            # Skip header and metadata lines
            if line.startswith(('Title', 'Authors', 'Table', '=', '-', 'Byte', 'Note', ' ')):
                continue
            if len(line.strip()) < 30:
                continue

            try:
                # Parse fixed-width columns
                name = line[0:12].strip()
                if not name:
                    continue

                # Reff (kpc) - Bytes 27-31
                reff_str = line[26:31].strip()
                reff = float(reff_str) if reff_str else None

                # VMag (Absolute V band magnitude) - Bytes 39-44
                vmag_str = line[38:44].strip()
                vmag_abs = float(vmag_str) if vmag_str else None

                # Sersic index - Bytes 46-50
                n_str = line[45:50].strip()
                n = float(n_str) if n_str else None

                # Ellipticity - Bytes 52-55
                e_str = line[51:55].strip()
                e = float(e_str) if e_str else 0.0

                # Position angle - Bytes 57-62
                pa_str = line[56:62].strip()
                pa = float(pa_str) if pa_str else 0.0

                # Skip if essential parameters are missing
                if reff is None or vmag_abs is None or n is None:
                    continue

                # Clamp Sersic index if needed
                if n > MAX_SERSIC_INDEX:
                    logger.debug(f"Galaxy {name}: Clamping n={n} to {MAX_SERSIC_INDEX}")
                    n = MAX_SERSIC_INDEX

                # Initialize galaxy entry if new
                if name not in galaxies:
                    galaxies[name] = []

                # Add component
                component = SersicComponent(
                    r_eff_kpc=reff,
                    abs_mag=vmag_abs,
                    n=n,
                    ellipticity=e,
                    pa_deg=pa
                )

                galaxies[name].append(component)

            except (ValueError, IndexError):
                # Skip malformed lines
                continue

    logger.debug(f"Parsed {len(galaxies)} galaxies from {filepath}")
    return galaxies


# =============================================================================
# Section 7: Config File Loader
# =============================================================================

def load_image_configs(
    filepath: str,
    profit_cli_override: Optional[str] = None
) -> List[ImageConfig]:
    """
    Load image generation configurations from a config file.

    The config file format:
    ```yaml
    image_configs:
      - name: "clean"
        pixel_scale: 0.3
        zeropoint: 27.0

      - name: "with_psf"
        pixel_scale: 0.3
        psf_enabled: true
        psf_fwhm: 1.0
    ```

    Parameters
    ----------
    filepath : str
        Path to config file (.yaml or .json)
    profit_cli_override : str, optional
        Override profit-cli path from CLI

    Returns
    -------
    list
        List of ImageConfig objects
    """
    data = load_file(filepath)

    # Handle both formats: with 'image_configs' key or direct list
    if isinstance(data, dict):
        config_list = data.get('image_configs', data.get('configs', [data]))
    else:
        config_list = data

    image_configs = []
    for cfg_dict in config_list:
        # CLI override takes precedence over config file
        profit_cli_path = profit_cli_override or cfg_dict.get('profit_cli_path')

        image_configs.append(ImageConfig(
            name=cfg_dict.get('name', 'default'),
            pixel_scale=cfg_dict.get('pixel_scale', DEFAULT_PIXEL_SCALE),
            zeropoint=cfg_dict.get('zeropoint', DEFAULT_ZEROPOINT),
            size_factor=cfg_dict.get('size_factor', DEFAULT_SIZE_FACTOR),
            size_pixels=cfg_dict.get('size_pixels'),
            psf_enabled=cfg_dict.get('psf_enabled', False),
            psf_type=cfg_dict.get('psf_type', 'gaussian'),
            psf_fwhm=cfg_dict.get('psf_fwhm', DEFAULT_PSF_FWHM),
            psf_moffat_beta=cfg_dict.get('psf_moffat_beta', DEFAULT_MOFFAT_BETA),
            psf_file=cfg_dict.get('psf_file'),
            sky_enabled=cfg_dict.get('sky_enabled', False),
            sky_type=cfg_dict.get('sky_type', 'flat'),
            sky_level=cfg_dict.get('sky_level', 0.0),
            sky_coeffs=cfg_dict.get('sky_coeffs', [0.0]),
            noise_enabled=cfg_dict.get('noise_enabled', False),
            noise_sigma=cfg_dict.get('noise_sigma'),
            noise_snr=cfg_dict.get('noise_snr'),
            noise_seed=cfg_dict.get('noise_seed'),
            engine=cfg_dict.get('engine', 'auto'),
            profit_cli_path=profit_cli_path
        ))

    logger.info(f"Loaded {len(image_configs)} image configs from {filepath}")
    return image_configs


# =============================================================================
# Section 8: Output (FITS)
# =============================================================================

def save_fits(image: np.ndarray, metadata: dict, filepath: Union[str, Path]) -> None:
    """
    Save image as FITS file with metadata in header.

    Parameters
    ----------
    image : ndarray
        2D image array
    metadata : dict
        Image metadata
    filepath : str or Path
        Output file path
    """
    hdu = fits.PrimaryHDU(image.astype(np.float32))
    header = hdu.header

    # Basic info
    header['OBJECT'] = metadata.get('name', 'MOCK')
    header['REDSHIFT'] = metadata.get('redshift', 0.0)
    header['PIXSCALE'] = (metadata.get('pixel_scale', 0.3), 'arcsec/pixel')
    header['MAGZERO'] = (metadata.get('zeropoint', 27.0), 'Photometric zeropoint')
    header['ENGINE'] = metadata.get('engine', 'unknown')
    header['CFGNAME'] = metadata.get('config_name', 'default')

    # PSF info
    header['PSF'] = metadata.get('psf_enabled', False)
    if metadata.get('psf_enabled'):
        header['PSFTYPE'] = metadata.get('psf_type', '')
        header['PSFFWHM'] = (metadata.get('psf_fwhm', 0.0), 'arcsec')

    # Sky and noise
    header['SKY'] = metadata.get('sky_enabled', False)
    header['NOISE'] = metadata.get('noise_enabled', False)

    # Component info
    n_comp = len(metadata.get('components', []))
    header['NCOMP'] = n_comp

    for i, comp in enumerate(metadata.get('components', [])):
        idx = i + 1
        header[f'RE_KPC{idx}'] = (comp.get('r_eff_kpc', 0.0), f'Re kpc, comp {idx}')
        header[f'RE_AS{idx}'] = (comp.get('re_arcsec', 0.0), f'Re arcsec, comp {idx}')
        header[f'RE_PX{idx}'] = (comp.get('re_pix', 0.0), f'Re pixels, comp {idx}')
        header[f'ABSMAG{idx}'] = (comp.get('abs_mag', 0.0), f'Abs mag, comp {idx}')
        header[f'APPMAG{idx}'] = (comp.get('app_mag', 0.0), f'App mag, comp {idx}')
        header[f'SERSIC{idx}'] = (comp.get('n', 0.0), f'Sersic n, comp {idx}')
        header[f'ELLIP{idx}'] = (comp.get('ellipticity', 0.0), f'Ellipticity, comp {idx}')
        header[f'PA{idx}'] = (comp.get('pa_deg', 0.0), f'PA deg, comp {idx}')

    # Write file
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(filepath, overwrite=True)
    logger.debug(f"Saved {filepath}")


def save_npy(image: np.ndarray, metadata: dict, filepath: Union[str, Path]) -> None:
    """Save image as numpy file with metadata JSON sidecar."""
    np.save(filepath, image)

    # Save metadata as JSON sidecar
    meta_path = Path(filepath).with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# Section 8.5: Visualization Functions
# =============================================================================

def visualize_galaxy(
    input_path: Union[str, Path, np.ndarray],
    metadata: Optional[dict] = None,
    output_path: Optional[Union[str, Path]] = None,
    cmap: str = 'viridis',
    sigma_smooth: float = 2.0,
    n_contours: int = 8,
    figsize: Tuple[float, float] = (10, 10),
    dpi: int = 150,
    show: bool = False
) -> Optional[Path]:
    """
    Visualize a mock galaxy image with arcsinh scaling and smoothed contours.

    This function creates a publication-quality visualization of a mock galaxy
    image using perceptually uniform colormaps and arcsinh scaling to handle
    the wide dynamic range. Smoothed contours are overlaid to highlight the
    galaxy's morphology.

    Parameters
    ----------
    input_path : str, Path, or ndarray
        Either a path to a FITS file or a 2D numpy array containing the image
    metadata : dict, optional
        Image metadata dictionary (required if input_path is an array)
    output_path : str or Path, optional
        Output PNG file path. If None, uses the same prefix as the input FITS file
        with a .png extension
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis'). Recommended options:
        'viridis', 'magma', 'inferno', 'cividis', 'plasma'
    sigma_smooth : float, optional
        Gaussian smoothing sigma in pixels for contour generation (default: 2.0)
    n_contours : int, optional
        Number of contour levels to plot (default: 8)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10, 10))
    dpi : int, optional
        Output resolution in dots per inch (default: 150)
    show : bool, optional
        If True, display the plot interactively (default: False)

    Returns
    -------
    Path or None
        Path to the saved PNG file, or None if matplotlib is not available

    Examples
    --------
    >>> # Visualize a FITS file
    >>> visualize_galaxy('output/NGC_1399_clean.fits')

    >>> # Visualize with custom colormap
    >>> visualize_galaxy('output/IC_1459_clean.fits', cmap='magma')

    >>> # Visualize from array with metadata
    >>> visualize_galaxy(image, metadata=meta, output_path='output/galaxy.png')

    Notes
    -----
    - Uses arcsinh scaling to handle the wide dynamic range of galaxy images
    - Smoothing is applied only for contour generation, not to the displayed image
    - Contours are plotted at logarithmically spaced levels for better morphology visibility
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return None

    # Load image and metadata
    if isinstance(input_path, np.ndarray):
        image = input_path
        if metadata is None:
            raise ValueError("metadata must be provided when input_path is an array")
        meta = metadata
        fits_path = None
    else:
        fits_path = Path(input_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        with fits.open(fits_path) as hdul:
            image = hdul[0].data
            header = hdul[0].header

            # Extract metadata from header
            meta = {
                'name': header.get('OBJECT', 'Unknown'),
                'redshift': header.get('REDSHIFT', 0.0),
                'pixel_scale': header.get('PIXSCALE', DEFAULT_PIXEL_SCALE),
                'zeropoint': header.get('MAGZERO', DEFAULT_ZEROPOINT),
                'config_name': header.get('CFGNAME', 'default'),
            }

    # Determine output path
    if output_path is None:
        if fits_path is None:
            raise ValueError("output_path must be provided when input_path is an array")
        output_path = fits_path.with_suffix('.png')
    else:
        output_path = Path(output_path)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Apply arcsinh scaling for display
    # Use a scale factor that highlights structure while avoiding saturation
    scale = np.nanpercentile(image, 99.5) / 10.0
    scaled_image = np.arcsinh(image / scale)

    # Display image with arcsinh scaling
    im = ax.imshow(scaled_image, origin='lower', cmap=cmap, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('arcsinh(Flux)', fontsize=12)

    # Generate smoothed version for contours
    smoothed = gaussian_filter(image, sigma=sigma_smooth)
    smoothed_scaled = np.arcsinh(smoothed / scale)

    # Determine contour levels (logarithmically spaced in flux space)
    vmin = np.nanpercentile(smoothed_scaled, 10)
    vmax = np.nanpercentile(smoothed_scaled, 99.5)
    levels = np.linspace(vmin, vmax, n_contours)

    # Plot contours
    contours = ax.contour(
        smoothed_scaled,
        levels=levels,
        colors='white',
        alpha=0.4,
        linewidths=0.8,
        linestyles='solid'
    )

    # Set title with galaxy name
    galaxy_name = meta.get('name', 'Mock Galaxy')
    config_name = meta.get('config_name', '')
    if config_name and config_name != 'default':
        title = f"{galaxy_name} ({config_name})"
    else:
        title = galaxy_name

    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    # Add axis labels
    pixel_scale = meta.get('pixel_scale', DEFAULT_PIXEL_SCALE)
    ax.set_xlabel(f'X [pixels] ({pixel_scale:.2f}"/pix)', fontsize=12)
    ax.set_ylabel(f'Y [pixels] ({pixel_scale:.2f}"/pix)', fontsize=12)

    # Add grid
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


# =============================================================================
# Section 9: Batch Processing with Parallelization
# =============================================================================

def process_single_job(args: Tuple[MockGalaxy, ImageConfig, str, str]) -> str:
    """
    Process a single galaxy+config combination.

    Parameters
    ----------
    args : tuple
        (galaxy, config, output_dir, output_format)

    Returns
    -------
    str
        Output file path
    """
    galaxy, config, output_dir, output_format = args

    gen = MockImageGenerator(config)
    image, metadata = gen.generate(galaxy)

    # Generate output filename (replace spaces with underscores)
    suffix = '.fits' if output_format == 'fits' else '.npy'
    safe_name = galaxy.name.replace(' ', '_')
    safe_config = config.name.replace(' ', '_')
    outpath = Path(output_dir) / f"{safe_name}_{safe_config}{suffix}"

    if output_format == 'fits':
        save_fits(image, metadata, outpath)
    else:
        save_npy(image, metadata, outpath)

    return str(outpath)


def run_batch(
    galaxies: List[MockGalaxy],
    configs: List[ImageConfig],
    output_dir: str,
    output_format: str = 'fits',
    workers: Optional[int] = None
) -> List[str]:
    """
    Run batch processing with multiprocessing.

    Parameters
    ----------
    galaxies : list
        List of MockGalaxy objects
    configs : list
        List of ImageConfig objects
    output_dir : str
        Output directory
    output_format : str
        Output format ('fits' or 'npy')
    workers : int, optional
        Number of worker processes (default: CPU count)

    Returns
    -------
    list
        List of output file paths
    """
    if workers is None:
        workers = cpu_count()

    # Create all jobs
    jobs = [
        (galaxy, config, output_dir, output_format)
        for galaxy in galaxies
        for config in configs
    ]

    total_jobs = len(jobs)
    logger.info(f"Processing {total_jobs} jobs with {workers} workers")

    if workers == 1:
        # Sequential processing
        results = []
        for i, job in enumerate(jobs):
            result = process_single_job(job)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1}/{total_jobs}")
    else:
        # Parallel processing
        with Pool(workers) as pool:
            results = pool.map(process_single_job, jobs)

    logger.info(f"Completed {len(results)} images")
    return results


# =============================================================================
# Section 10: CLI Interface
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate mock galaxy images with Sersic profiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--models", "-m",
        metavar="FILE",
        help="Input Model File (YAML/JSON) containing galaxy definitions"
    )
    mode.add_argument(
        "--single", "-s",
        action="store_true",
        help="Single galaxy mode (specify parameters via CLI)"
    )

    # Config file (optional, works with --models)
    parser.add_argument(
        "--config", "-c",
        metavar="FILE",
        help="Config file (YAML/JSON) with image generation settings (PSF, sky, noise)"
    )

    # Single galaxy parameters
    single = parser.add_argument_group("Single Galaxy Parameters")
    single.add_argument("--name", default="mock_galaxy", help="Galaxy name")
    single.add_argument("-z", "--redshift", type=float, default=DEFAULT_REDSHIFT,
                        help="Redshift")
    single.add_argument("--r-eff", type=float, nargs="+", metavar="KPC",
                        help="Effective radius in kpc for each component")
    single.add_argument("--abs-mag", type=float, nargs="+", metavar="MAG",
                        help="Absolute magnitude for each component")
    single.add_argument("--sersic-n", type=float, nargs="+", metavar="N",
                        help="Sersic index for each component")
    single.add_argument("--ellip", type=float, nargs="+", default=[0.0],
                        help="Ellipticity for each component")
    single.add_argument("--pa", type=float, nargs="+", default=[0.0],
                        help="Position angle (degrees) for each component")

    # Image parameters
    img = parser.add_argument_group("Image Parameters")
    img.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE,
                     help="Pixel scale in arcsec/pixel")
    img.add_argument("--zeropoint", type=float, default=DEFAULT_ZEROPOINT,
                     help="Photometric zeropoint")
    img.add_argument("--size-factor", type=float, default=DEFAULT_SIZE_FACTOR,
                     help="Image half-size = factor * max(Re)")
    img.add_argument("--size", type=int, metavar="PIXELS",
                     help="Fixed image size in pixels (overrides size-factor)")

    # PSF parameters
    psf = parser.add_argument_group("PSF Parameters")
    psf.add_argument("--psf", action="store_true",
                     help="Enable PSF convolution")
    psf.add_argument("--psf-fwhm", type=float, default=DEFAULT_PSF_FWHM,
                     help="PSF FWHM in arcsec")
    psf.add_argument("--psf-type", choices=["gaussian", "moffat", "image"],
                     default="gaussian", help="PSF type")
    psf.add_argument("--psf-file", metavar="FILE",
                     help="PSF image file (for --psf-type image)")
    psf.add_argument("--moffat-beta", type=float, default=DEFAULT_MOFFAT_BETA,
                     help="Moffat beta parameter")

    # Sky parameters
    sky = parser.add_argument_group("Sky Background")
    sky.add_argument("--sky", type=float, metavar="LEVEL",
                     help="Flat sky level (enables sky background)")
    sky.add_argument("--sky-tilted", type=float, nargs="+", metavar="COEFF",
                     help="Tilted sky polynomial coefficients [a, b, c, d, e, f]")

    # Noise parameters
    noise = parser.add_argument_group("Noise")
    noise.add_argument("--noise-sigma", type=float, metavar="SIGMA",
                       help="Gaussian noise sigma (enables noise)")
    noise.add_argument("--snr", type=float, metavar="SNR",
                       help="Target S/N at effective radius (enables noise)")
    noise.add_argument("--seed", type=int, help="Random seed for noise")

    # Engine selection
    engine_group = parser.add_argument_group("Engine")
    engine_group.add_argument("--engine", choices=["libprofit", "astropy", "auto"],
                              default="auto", help="Sersic rendering engine")
    engine_group.add_argument("--profit-cli", metavar="PATH",
                              help="Path to profit-cli binary (or directory containing it)")

    # Output parameters
    out = parser.add_argument_group("Output")
    out.add_argument("-o", "--output", default=".", metavar="DIR",
                     help="Output directory")
    out.add_argument("--format", choices=["fits", "npy"], default="fits",
                     help="Output file format")

    # Galaxy selection (for --models mode)
    parser.add_argument("--galaxy", nargs="+", metavar="NAME",
                        help="Select specific galaxies from model file")

    # Parallelization
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Number of worker processes (default: {cpu_count()})")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get profit-cli path from CLI argument
    profit_cli = getattr(args, 'profit_cli', None)

    # Build image config from CLI args
    def build_image_config(name: str = "cli") -> ImageConfig:
        return ImageConfig(
            name=name,
            pixel_scale=args.pixel_scale,
            zeropoint=args.zeropoint,
            size_factor=args.size_factor,
            size_pixels=args.size,
            psf_enabled=args.psf,
            psf_type=args.psf_type,
            psf_fwhm=args.psf_fwhm,
            psf_moffat_beta=args.moffat_beta,
            psf_file=args.psf_file,
            sky_enabled=(args.sky is not None or args.sky_tilted is not None),
            sky_type="tilted" if args.sky_tilted else "flat",
            sky_level=args.sky if args.sky else 0.0,
            sky_coeffs=args.sky_tilted if args.sky_tilted else [0.0],
            noise_enabled=(args.noise_sigma is not None or args.snr is not None),
            noise_sigma=args.noise_sigma,
            noise_snr=args.snr,
            noise_seed=args.seed,
            engine=args.engine,
            profit_cli_path=profit_cli
        )

    # Process based on mode
    if args.models:
        # Model file mode
        galaxies = load_model_file(args.models, args.galaxy)

        if not galaxies:
            logger.error("No galaxies to process")
            sys.exit(1)

        # Load image configs from config file or build from CLI
        if args.config:
            image_configs = load_image_configs(args.config, profit_cli_override=profit_cli)
        else:
            image_configs = [build_image_config()]

        results = run_batch(galaxies, image_configs, args.output, args.format, args.workers)
        print(f"Generated {len(results)} images")

    else:
        # Single galaxy mode
        if not all([args.r_eff, args.abs_mag, args.sersic_n]):
            parser.error("--r-eff, --abs-mag, and --sersic-n required for single mode")

        n_comp = len(args.r_eff)
        if len(args.abs_mag) != n_comp or len(args.sersic_n) != n_comp:
            parser.error("--r-eff, --abs-mag, and --sersic-n must have same length")

        # Extend ellip and pa to match number of components
        ellip = args.ellip + [args.ellip[-1]] * (n_comp - len(args.ellip))
        pa = args.pa + [args.pa[-1]] * (n_comp - len(args.pa))

        components = [
            SersicComponent(
                r_eff_kpc=args.r_eff[i],
                abs_mag=args.abs_mag[i],
                n=args.sersic_n[i],
                ellipticity=ellip[i],
                pa_deg=pa[i]
            )
            for i in range(n_comp)
        ]

        galaxy = MockGalaxy(
            name=args.name,
            redshift=args.redshift,
            components=components
        )

        image_config = build_image_config()
        gen = MockImageGenerator(image_config)
        image, metadata = gen.generate(galaxy)

        # Save output (replace spaces with underscores in filename)
        suffix = '.fits' if args.format == 'fits' else '.npy'
        safe_name = args.name.replace(' ', '_')
        outpath = Path(args.output) / f"{safe_name}{suffix}"

        if args.format == 'fits':
            save_fits(image, metadata, outpath)
        else:
            save_npy(image, metadata, outpath)

        print(f"Generated {outpath}")
        print(f"Image size: {image.shape}")
        print(f"Max pixel value: {image.max():.2e}")


if __name__ == "__main__":
    main()
