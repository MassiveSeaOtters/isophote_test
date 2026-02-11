# MockGal Benchmark Results

## System Information

**Date:** 2026-02-11 15:57:45

- **Platform:** macOS-15.7.3-arm64-arm-64bit
- **Os:** Darwin
- **Os Version:** Darwin Kernel Version 24.6.0: Wed Nov  5 21:28:03 PST 2025; root:xnu-11417.140.69.705.2~1/RELEASE_ARM64_T8122
- **Architecture:** arm64
- **Processor:** arm
- **Python Version:** 3.12.11 | packaged by conda-forge | (main, Jun  4 2025, 14:38:53) [Clang 18.1.8 ]
- **Python Implementation:** CPython
- **Total Memory Gb:** 24.0
- **Available Memory Gb:** 5.17
- **Cpu Count:** 8
- **Cpu Count Logical:** 8

## Rendering Speed Summary

| Sersic Index | Image Size | Astropy (ms) | Libprofit (ms) | Faster |
|--------------|------------|--------------|----------------|--------|
| 1.0 | 256 | 1.15 | 25.45 | astropy (22.2x) |
| 1.0 | 512 | 5.92 | 79.06 | astropy (13.4x) |
| 1.0 | 1024 | 26.86 | 283.85 | astropy (10.6x) |
| 2.0 | 256 | 1.12 | 29.81 | astropy (26.7x) |
| 2.0 | 512 | 5.59 | 81.07 | astropy (14.5x) |
| 2.0 | 1024 | 23.33 | 279.87 | astropy (12.0x) |
| 4.0 | 256 | 1.12 | 27.56 | astropy (24.6x) |
| 4.0 | 512 | 4.95 | 80.04 | astropy (16.2x) |
| 4.0 | 1024 | 22.38 | 273.50 | astropy (12.2x) |
| 6.0 | 256 | 1.09 | 32.55 | astropy (29.9x) |
| 6.0 | 512 | 4.97 | 83.45 | astropy (16.8x) |
| 6.0 | 1024 | 21.92 | 284.71 | astropy (13.0x) |
| 8.0 | 256 | 1.10 | 28.97 | astropy (26.4x) |
| 8.0 | 512 | 5.48 | 78.92 | astropy (14.4x) |
| 8.0 | 1024 | 22.71 | 274.27 | astropy (12.1x) |

## PSF Convolution Overhead

| PSF Size | Engine | Time (ms) | Overhead |
|----------|--------|-----------|----------|
| 11x11 | astropy | 30.50 | 495% |
| 33x33 | astropy | 307.59 | 5897% |
| 65x65 | astropy | 1268.39 | 24631% |
| 11x11 | libprofit | 78.82 | -4% |
| 33x33 | libprofit | 77.47 | -5% |
| 65x65 | libprofit | 78.53 | -4% |

## Profile Consistency

Comparison against point-evaluation Sersic formula. **Note:** This test favors astropy's 
point-evaluation method. Libprofit uses pixel integration which is more physically accurate 
but shows "deviation" from simple point formulas.

| Sersic Index | Engine | Max Deviation | Median Deviation | Method |
|--------------|--------|---------------|------------------|--------|
| 1.0 | astropy | 0.0000 | 0.0000 | Point eval |
| 1.0 | libprofit | 0.0053 | 0.0003 | Pixel integration |
| 4.0 | astropy | 0.0000 | 0.0000 | Point eval |
| 4.0 | libprofit | 0.0715 | 0.0080 | Pixel integration |
| 8.0 | astropy | 0.0000 | 0.0000 | Point eval |
| 8.0 | libprofit | 0.0987 | 0.0083 | Pixel integration |

## Notes

- Complete data available in `benchmark_results.json`
- Times are averaged over multiple iterations
- "Deviation" measures consistency with point-evaluation formula, not accuracy
- Libprofit's pixel integration is more accurate physically but differs from point formulas
