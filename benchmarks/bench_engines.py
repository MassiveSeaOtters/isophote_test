#!/usr/bin/env python
"""
Benchmark comparing pyprofit and astropy Sersic rendering engines.

Run with: python benchmarks/bench_engines.py

Tests:
1. Rendering speed for various Sersic indices and image sizes
2. Accuracy comparison against analytical profile
3. PSF convolution performance
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaincinv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mockgal import SersicEngine, HAS_PYPROFIT


def benchmark_rendering_speed(n_iterations: int = 10):
    """
    Benchmark rendering speed for different parameters.

    Returns
    -------
    list of dict
        Results with timing information
    """
    print("=" * 60)
    print("Benchmark: Rendering Speed")
    print("=" * 60)

    results = []

    sersic_indices = [1.0, 2.0, 4.0, 6.0, 8.0]
    image_sizes = [256, 512, 1024]
    engines_to_test = ["astropy"]

    if HAS_PYPROFIT:
        engines_to_test.append("pyprofit")
    else:
        print("Note: pyprofit not available, benchmarking astropy only")

    for n in sersic_indices:
        for size in image_sizes:
            row = {'n': n, 'size': size}

            for engine_name in engines_to_test:
                try:
                    engine = SersicEngine(engine_name)

                    # Warm-up run
                    engine.render(
                        shape=(size, size),
                        xcen=size/2, ycen=size/2,
                        mag=15.0, re_pix=size/10,
                        n=n, axrat=0.7, ang=30.0,
                        zeropoint=27.0
                    )

                    # Timed runs
                    start = time.perf_counter()
                    for _ in range(n_iterations):
                        engine.render(
                            shape=(size, size),
                            xcen=size/2, ycen=size/2,
                            mag=15.0, re_pix=size/10,
                            n=n, axrat=0.7, ang=30.0,
                            zeropoint=27.0
                        )
                    elapsed = (time.perf_counter() - start) / n_iterations

                    row[f'{engine_name}_s'] = elapsed

                except Exception as e:
                    print(f"  Error with {engine_name}: {e}")
                    row[f'{engine_name}_s'] = None

            # Compute speedup if both available
            if HAS_PYPROFIT and row.get('pyprofit_s') and row.get('astropy_s'):
                row['speedup'] = row['astropy_s'] / row['pyprofit_s']
            else:
                row['speedup'] = None

            results.append(row)

            # Print result
            astropy_time = row.get('astropy_s', 0) * 1000
            pyprofit_time = row.get('pyprofit_s', 0) * 1000 if row.get('pyprofit_s') else None
            speedup = row.get('speedup', '')

            if pyprofit_time is not None:
                print(f"n={n:.1f}, size={size:4d}: "
                      f"astropy={astropy_time:6.2f}ms, "
                      f"pyprofit={pyprofit_time:6.2f}ms, "
                      f"speedup={speedup:.1f}x")
            else:
                print(f"n={n:.1f}, size={size:4d}: astropy={astropy_time:6.2f}ms")

    return results


def benchmark_accuracy():
    """
    Compare rendered profiles against analytical 1D Sersic profile.

    Returns
    -------
    list of dict
        Results with accuracy metrics
    """
    print("\n" + "=" * 60)
    print("Benchmark: Accuracy vs Analytical Profile")
    print("=" * 60)

    results = []

    sersic_indices = [1.0, 4.0, 8.0]
    engines_to_test = ["astropy"]

    if HAS_PYPROFIT:
        engines_to_test.append("pyprofit")

    size = 501
    re_pix = 50.0

    for n in sersic_indices:
        for engine_name in engines_to_test:
            try:
                engine = SersicEngine(engine_name)

                image = engine.render(
                    shape=(size, size),
                    xcen=size//2, ycen=size//2,
                    mag=15.0, re_pix=re_pix,
                    n=n, axrat=1.0, ang=0.0,  # Circular
                    zeropoint=27.0
                )

                # Extract radial profile along +X axis
                center = size // 2
                r = np.arange(1, 200)
                profile = image[center, center + r]

                # Compute analytical profile
                # I(r) = I_e * exp(-b_n * ((r/Re)^(1/n) - 1))
                b_n = gammaincinv(2 * n, 0.5)
                I_e = profile[int(re_pix) - 1]  # Value at Re
                analytical = I_e * np.exp(-b_n * ((r / re_pix) ** (1/n) - 1))

                # Compute deviation (skip very center and far edges)
                mask = (r > 5) & (r < 150)
                rel_dev = np.abs(profile[mask] - analytical[mask]) / analytical[mask]

                max_dev = float(np.nanmax(rel_dev))
                median_dev = float(np.nanmedian(rel_dev))

                results.append({
                    'n': n,
                    'engine': engine_name,
                    'max_rel_dev': max_dev,
                    'median_rel_dev': median_dev
                })

                print(f"n={n:.1f}, {engine_name:10s}: "
                      f"max_dev={max_dev:.4f}, median_dev={median_dev:.4f}")

            except Exception as e:
                print(f"  Error with n={n}, {engine_name}: {e}")

    return results


def benchmark_psf_convolution(n_iterations: int = 5):
    """
    Benchmark PSF convolution overhead.

    Returns
    -------
    list of dict
        Results with timing information
    """
    print("\n" + "=" * 60)
    print("Benchmark: PSF Convolution Overhead")
    print("=" * 60)

    results = []
    engines_to_test = ["astropy"]

    if HAS_PYPROFIT:
        engines_to_test.append("pyprofit")

    size = 512
    psf_sizes = [11, 33, 65]  # Odd sizes for centered PSF

    for engine_name in engines_to_test:
        engine = SersicEngine(engine_name)

        # Baseline: no PSF
        start = time.perf_counter()
        for _ in range(n_iterations):
            engine.render(
                shape=(size, size),
                xcen=size/2, ycen=size/2,
                mag=15.0, re_pix=50.0,
                n=4.0, axrat=0.7, ang=30.0,
                zeropoint=27.0,
                psf=None
            )
        baseline = (time.perf_counter() - start) / n_iterations

        for psf_size in psf_sizes:
            # Create Gaussian PSF
            y, x = np.mgrid[:psf_size, :psf_size] - psf_size // 2
            sigma = psf_size / 6
            psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            psf = psf / psf.sum()

            start = time.perf_counter()
            for _ in range(n_iterations):
                engine.render(
                    shape=(size, size),
                    xcen=size/2, ycen=size/2,
                    mag=15.0, re_pix=50.0,
                    n=4.0, axrat=0.7, ang=30.0,
                    zeropoint=27.0,
                    psf=psf
                )
            with_psf = (time.perf_counter() - start) / n_iterations

            overhead = (with_psf - baseline) / baseline * 100

            results.append({
                'engine': engine_name,
                'psf_size': psf_size,
                'baseline_ms': baseline * 1000,
                'with_psf_ms': with_psf * 1000,
                'overhead_pct': overhead
            })

            print(f"{engine_name:10s}, PSF {psf_size:2d}x{psf_size:2d}: "
                  f"baseline={baseline*1000:.2f}ms, "
                  f"with_psf={with_psf*1000:.2f}ms, "
                  f"overhead={overhead:.1f}%")

    return results


def benchmark_ellipticity_range():
    """
    Test rendering at various ellipticities.

    Returns
    -------
    list of dict
        Results showing successful rendering at each ellipticity
    """
    print("\n" + "=" * 60)
    print("Benchmark: Ellipticity Range")
    print("=" * 60)

    results = []
    engines_to_test = ["astropy"]

    if HAS_PYPROFIT:
        engines_to_test.append("pyprofit")

    ellipticities = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    size = 201

    for engine_name in engines_to_test:
        engine = SersicEngine(engine_name)

        for ellip in ellipticities:
            axrat = 1 - ellip

            try:
                start = time.perf_counter()
                image = engine.render(
                    shape=(size, size),
                    xcen=size/2, ycen=size/2,
                    mag=15.0, re_pix=30.0,
                    n=4.0, axrat=axrat, ang=45.0,
                    zeropoint=27.0
                )
                elapsed = time.perf_counter() - start

                # Verify result is valid
                is_valid = (
                    np.isfinite(image).all() and
                    image.max() > 0 and
                    image.min() >= 0
                )

                results.append({
                    'engine': engine_name,
                    'ellipticity': ellip,
                    'axrat': axrat,
                    'valid': is_valid,
                    'time_ms': elapsed * 1000
                })

                status = "OK" if is_valid else "FAIL"
                print(f"{engine_name:10s}, ellip={ellip:.2f} (b/a={axrat:.2f}): "
                      f"{status}, {elapsed*1000:.2f}ms")

            except Exception as e:
                results.append({
                    'engine': engine_name,
                    'ellipticity': ellip,
                    'axrat': axrat,
                    'valid': False,
                    'error': str(e)
                })
                print(f"{engine_name:10s}, ellip={ellip:.2f}: ERROR - {e}")

    return results


def main():
    """Run all benchmarks."""
    print("Mock Galaxy Image Generator - Engine Benchmarks")
    print(f"pyprofit available: {HAS_PYPROFIT}")
    print()

    all_results = {}

    # Run benchmarks
    all_results['rendering_speed'] = benchmark_rendering_speed()
    all_results['accuracy'] = benchmark_accuracy()
    all_results['psf_convolution'] = benchmark_psf_convolution()
    all_results['ellipticity_range'] = benchmark_ellipticity_range()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if HAS_PYPROFIT:
        speed_results = all_results['rendering_speed']
        speedups = [r['speedup'] for r in speed_results if r.get('speedup')]
        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"Average pyprofit speedup over astropy: {avg_speedup:.1f}x")
    else:
        print("Install pyprofit for performance comparison")

    # Save results to JSON
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
