# Benchmarks Directory

Performance benchmarks for MockGal rendering engines and operations.

## Benchmark Types

### 1. Rendering Speed
Tests rendering time for different Sersic indices and image sizes using Astropy backend.

**Metrics:**
- Sersic index: n = 0.5 to 8.0
- Image sizes: 101, 501, 1001, 2001 pixels
- Output: Time in seconds per render

### 2. Accuracy Comparison
Compares flux conservation and profile accuracy between backends when LibProfit is available.

**Metrics:**
- Total flux difference between Astropy and LibProfit
- Radial profile comparison
- Relative errors

### 3. PSF Convolution Overhead
Measures the performance impact of PSF convolution.

**Metrics:**
- Time without PSF
- Time with PSF (various sizes: 15, 31, 51 pixels)
- Overhead percentage

### 4. Ellipticity Range
Tests rendering across different ellipticities (axis ratios).

**Metrics:**
- Ellipticity: 0.0 to 0.7
- Rendering time vs. ellipticity
- Flux conservation check

## Running Benchmarks

Execute all benchmarks:
```bash
python benchmarks/bench_engines.py
```

The script will:
1. Collect system information (CPU, memory, OS, Python version)
2. Run all benchmark suites
3. Save results to two files:
   - `benchmark_results.json` - Complete data with system info
   - `benchmark_results.md` - Human-readable summary report

## Output Files

### benchmark_results.json

Complete benchmark data in JSON format including:
- System information (platform, CPU, memory, Python version)
- Timestamp of benchmark run
- Raw benchmark results for all tests
- Useful for programmatic analysis and comparison

Example structure:
```json
{
  "system_info": {
    "platform": "macOS-14.0-arm64",
    "processor": "arm",
    "total_memory_gb": 16.0,
    "cpu_count": 8
  },
  "timestamp": "2026-01-29 10:30:45",
  "rendering_speed": [...],
  "accuracy": [...],
  "psf_convolution": [...],
  "ellipticity_range": [...]
}
```

### benchmark_results.md

Human-readable markdown summary with:
- System information table
- Rendering speed summary table
- PSF overhead summary table
- Notes and interpretation guidelines

## System Requirements

**Minimum:**
- Python 3.8+
- NumPy, Astropy
- 4 GB RAM

**Recommended:**
- LibProfit compiled and available in PATH
- 8+ GB RAM for large image benchmarks
- psutil package for detailed system info

**Optional:**
- psutil - For detailed memory and CPU information
  ```bash
  pip install psutil
  ```

## Interpreting Results

### Rendering Speed
- Times increase with Sersic index (higher n = more complex profile)
- Times scale approximately with image area (quadratic in size)
- LibProfit typically 5-10x faster than Astropy for large images

### PSF Overhead
- Overhead is primarily from FFT convolution
- Larger PSF kernels add minimal overhead (FFT is efficient)
- Expected overhead: 20-50% for typical PSF sizes

### Accuracy
- Flux should be conserved to within 1-2% between backends
- Profile differences typically < 5% in outer regions
- Larger differences may indicate numerical issues

## Benchmarking Best Practices

1. **Close other applications** - Minimize background processes
2. **Run multiple times** - Results can vary ±10% between runs
3. **Check thermals** - CPU throttling affects performance
4. **Document system state** - Note any unusual conditions
5. **Version control** - Track benchmark results with git tags

## Comparing Results Across Systems

The system_info field enables fair comparison:
- Same architecture (ARM vs x86) can differ 2-3x
- Memory bandwidth affects large image performance
- Python implementation (CPython vs PyPy) matters
- Always compare within same major Python version

## Future Benchmark Ideas

- Memory usage profiling (peak RAM per operation)
- Multi-threading scalability tests
- Batch processing throughput
- GPU acceleration benchmarks (if implemented)
- Cache efficiency analysis
