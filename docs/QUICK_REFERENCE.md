# MockGal Quick Reference

## Recent Session (2026-02-11)

**Status:** Repository reorganized, mock generation system added, ready to push

**Git:** 1 commit ahead of origin/main (commit 91b4123)

## Essential Commands

### Run Tests
```bash
pytest tests/test_mockgal.py -v              # Full suite (62 tests)
pytest tests/test_mockgal.py::TestVisualization -v  # Specific class
```

### Run Benchmarks
```bash
python benchmarks/bench_engines.py           # Generates JSON + MD reports
```

### Generate Mocks
```bash
# Test with 2 galaxies
python inputs/generate_huang2013_mocks.py --output /path --test

# Full dataset (93 galaxies, ~35 min, ~23 GB)
python inputs/generate_huang2013_mocks.py --output /path

# Specific galaxies
python inputs/generate_huang2013_mocks.py --output /path \
    --galaxies "NGC 3923" "IC 1459"
```

## Critical Conventions

### Filenames (ALWAYS use sanitize_filename())
```python
from mockgal import sanitize_filename

# Correct
filename = f"{sanitize_filename(galaxy_name)}_{config_name}.fits"
# Result: "NGC3923_clean.fits"

# Wrong (old convention)
filename = f"{galaxy_name.replace(' ', '_')}_{config_name}.fits"
# Result: "NGC_3923_clean.fits"  ❌
```

### File Organization
- `inputs/`: Models, configs, scripts
- `tests/`: Test suite
- `benchmarks/`: Performance tests
- `output/`: Generated images (not in git)

## Breaking Changes (Need Migration)

1. **Folder:** `examples/` → `inputs/`
2. **Filenames:** `NGC_3923` → `NGC3923`

See `MIGRATION.md` for complete migration guide.

## Mock Configurations

| Mock | z | Noise | Purpose |
|------|---|-------|---------|
| mock1 | 0.05 | No | Ground truth baseline |
| mock2 | 0.05 | Yes | Realistic nearby |
| mock3 | 0.20 | Yes | Intermediate redshift |
| mock4 | 0.50 | Yes | High-z challenging |

All use: HSC pixel scale (0.168"/pix), FWHM=0.7"

## Important Notes

### Benchmark Interpretation
- "Profile Consistency" is NOT accuracy measurement
- Astropy shows 0.0000 deviation (point-evaluation bias)
- Libprofit shows deviation (pixel integration, MORE accurate)

### Performance
- Astropy: 11-29× faster (small images, no PSF)
- Libprofit: Better for large images with PSF

## Documentation Files

- `CLAUDE.md`: Project conventions and ground rules
- `MIGRATION.md`: Breaking changes guide
- `inputs/README.md`: Input files and mock generation
- `tests/README.md`: Test suite documentation
- `benchmarks/README.md`: Benchmark interpretation
- `output/README.md`: Output organization
- `docs/SESSION_2026-02-11.md`: Detailed session notes

## Next Steps

**Ready to push:**
```bash
git push  # Push commit 91b4123
```

**Potential improvements:**
- Add progress bar to mock generation
- Consider parallel mock generation
- Add PNG generation option to mock script

## Contact / Issues

For detailed history: See `docs/SESSION_2026-02-11.md`
For conventions: See `CLAUDE.md`
For migration help: See `MIGRATION.md`
