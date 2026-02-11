#!/usr/bin/env python
"""
Benchmark comparing libprofit and astropy Sersic rendering engines.

Run with: python benchmarks/bench_engines.py

Tests:
1. Rendering speed for various Sersic indices and image sizes
2. Accuracy comparison against analytical profile
3. PSF convolution performance
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaincinv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mockgal import SersicEngine

# Check if libprofit is available
try:
    from mockgal import find_profit_cli
    HAS_LIBPROFIT = find_profit_cli() is not None
except (ImportError, Exception):
    HAS_LIBPROFIT = False


def get_system_info() -> dict:
    """
    Collect system information for benchmark context.

    Returns
    -------
    dict
        System information including CPU, memory, OS, Python version
    """
    info = {
        'platform': platform.platform(),
        'os': platform.system(),
        'os_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'python_implementation': platform.python_implementation(),
    }

    # Add memory info if psutil available (optional dependency)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_memory_gb'] = round(mem.total / (1024**3), 2)
        info['available_memory_gb'] = round(mem.available / (1024**3), 2)
        info['cpu_count'] = psutil.cpu_count(logical=False)
        info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    except ImportError:
        info['memory_note'] = 'psutil not available (install for detailed memory info)'

    return info


def generate_summary_report(results: dict, output_path: Path) -> None:
    """
    Generate human-readable markdown summary of benchmark results.

    Parameters
    ----------
    results : dict
        Benchmark results dictionary with system_info and benchmark data
    output_path : Path
        Output markdown file path
    """
    with open(output_path, 'w') as f:
        f.write("# MockGal Benchmark Results\n\n")

        # System info
        f.write("## System Information\n\n")
        f.write(f"**Date:** {results['timestamp']}\n\n")
        for key, value in results['system_info'].items():
            f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")

        # Rendering speed summary
        f.write("\n## Rendering Speed Summary\n\n")
        f.write("| Sersic Index | Image Size | Astropy (ms) | Libprofit (ms) | Faster |\n")
        f.write("|--------------|------------|--------------|----------------|--------|\n")
        for entry in results['rendering_speed']:
            astropy_ms = entry['astropy_s'] * 1000
            libprofit_ms = entry.get('libprofit_s', 0) * 1000 if entry.get('libprofit_s') else None

            if libprofit_ms:
                # Calculate which is faster
                if astropy_ms < libprofit_ms:
                    ratio = libprofit_ms / astropy_ms
                    faster = f"astropy ({ratio:.1f}x)"
                else:
                    ratio = astropy_ms / libprofit_ms
                    faster = f"libprofit ({ratio:.1f}x)"
                f.write(f"| {entry['n']:.1f} | {entry['size']} | {astropy_ms:.2f} | {libprofit_ms:.2f} | {faster} |\n")
            else:
                f.write(f"| {entry['n']:.1f} | {entry['size']} | {astropy_ms:.2f} | - | - |\n")

        # PSF overhead summary
        f.write("\n## PSF Convolution Overhead\n\n")
        f.write("| PSF Size | Engine | Time (ms) | Overhead |\n")
        f.write("|----------|--------|-----------|----------|\n")
        for entry in results['psf_convolution']:
            time_ms = entry['with_psf_ms']
            overhead = entry['overhead_pct']
            f.write(f"| {entry['psf_size']}x{entry['psf_size']} | {entry['engine']} | {time_ms:.2f} | {overhead:.0f}% |\n")

        # Profile consistency summary
        f.write("\n## Profile Consistency\n\n")
        f.write("Comparison against point-evaluation Sersic formula. **Note:** This test favors astropy's \n")
        f.write("point-evaluation method. Libprofit uses pixel integration which is more physically accurate \n")
        f.write("but shows \"deviation\" from simple point formulas.\n\n")
        f.write("| Sersic Index | Engine | Max Deviation | Median Deviation | Method |\n")
        f.write("|--------------|--------|---------------|------------------|--------|\n")
        for entry in results['accuracy']:
            method = "Point eval" if entry['engine'] == 'astropy' else "Pixel integration"
            f.write(f"| {entry['n']:.1f} | {entry['engine']} | {entry['max_rel_dev']:.4f} | {entry['median_rel_dev']:.4f} | {method} |\n")

        f.write("\n## Notes\n\n")
        f.write("- Complete data available in `benchmark_results.json`\n")
        f.write("- Times are averaged over multiple iterations\n")
        f.write("- \"Deviation\" measures consistency with point-evaluation formula, not accuracy\n")
        f.write("- Libprofit's pixel integration is more accurate physically but differs from point formulas\n")


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

    if HAS_LIBPROFIT:
        engines_to_test.append("libprofit")
    else:
        print("Note: libprofit not available, benchmarking astropy only")

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
            if HAS_LIBPROFIT and row.get('libprofit_s') and row.get('astropy_s'):
                row['speedup'] = row['astropy_s'] / row['libprofit_s']
            else:
                row['speedup'] = None

            results.append(row)

            # Print result
            astropy_time = row.get('astropy_s', 0) * 1000
            libprofit_time = row.get('libprofit_s', 0) * 1000 if row.get('libprofit_s') else None
            speedup = row.get('speedup', '')

            if libprofit_time is not None:
                print(f"n={n:.1f}, size={size:4d}: "
                      f"astropy={astropy_time:6.2f}ms, "
                      f"libprofit={libprofit_time:6.2f}ms, "
                      f"speedup={speedup:.1f}x")
            else:
                print(f"n={n:.1f}, size={size:4d}: astropy={astropy_time:6.2f}ms")

    return results


def benchmark_accuracy():
    """
    Compare rendering methods between engines.

    Note: This compares against a point-evaluation Sersic formula, which
    favors astropy's point-evaluation method. Libprofit uses pixel integration
    which is more physically accurate but shows deviation from point formulas.

    Returns
    -------
    list of dict
        Results with consistency metrics
    """
    print("\n" + "=" * 60)
    print("Benchmark: Profile Consistency (Point-Evaluation Formula)")
    print("=" * 60)

    results = []

    sersic_indices = [1.0, 4.0, 8.0]
    engines_to_test = ["astropy"]

    if HAS_LIBPROFIT:
        engines_to_test.append("libprofit")

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

    if HAS_LIBPROFIT:
        engines_to_test.append("libprofit")

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

    if HAS_LIBPROFIT:
        engines_to_test.append("libprofit")

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
                is_valid = bool(
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
    """Run all benchmarks and save results."""
    print("="*60)
    print("MockGal Benchmark Suite")
    print("="*60)

    print("\nCollecting system information...")
    system_info = get_system_info()

    print("\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    print(f"\nlibprofit available: {HAS_LIBPROFIT}")
    print()

    # Run benchmarks
    speed_results = benchmark_rendering_speed()
    accuracy_results = benchmark_accuracy()
    psf_results = benchmark_psf_convolution()
    ellipticity_results = benchmark_ellipticity_range()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if HAS_LIBPROFIT:
        speedups = [r['speedup'] for r in speed_results if r.get('speedup')]
        if speedups:
            avg_ratio = np.mean(speedups)
            if avg_ratio < 1.0:
                # Astropy is faster
                print(f"Average performance: astropy is {1/avg_ratio:.1f}x faster than libprofit")
            else:
                # Libprofit is faster
                print(f"Average performance: libprofit is {avg_ratio:.1f}x faster than astropy")
    else:
        print("Install libprofit for performance comparison")

    # Save results with system info and timestamp
    results = {
        'system_info': system_info,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'rendering_speed': speed_results,
        'accuracy': accuracy_results,
        'psf_convolution': psf_results,
        'ellipticity_range': ellipticity_results,
    }

    output_file = Path(__file__).parent / 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Generate summary report
    summary_file = output_file.with_suffix('.md')
    generate_summary_report(results, summary_file)
    print(f"Summary report saved to {summary_file}")


if __name__ == "__main__":
    main()
