# Migration Guide: MockGal v2.0

This guide helps you migrate scripts and workflows to MockGal v2.0.

## Breaking Changes

### 1. Folder Rename: `examples/` → `inputs/`

**Reason:** The "examples" folder primarily contained input data (models, configs) rather than example code, causing semantic confusion.

**Impact:** All command-line arguments and script paths that reference `examples/` must be updated.

#### Before (v1.x)
```bash
python mockgal.py \
    --models examples/huang2013_models.yaml \
    --config examples/example_image_config.yaml \
    -o output/
```

#### After (v2.0)
```bash
python mockgal.py \
    --models inputs/huang2013_models.yaml \
    --config inputs/example_image_config.yaml \
    -o output/
```

#### Migration Commands

**Update all Python scripts:**
```bash
# Review changes first
grep -r "examples/" --include="*.py" .

# Replace (dry run with confirmation)
find . -name "*.py" -exec sed -i.bak "s|examples/|inputs/|g" {} \;

# Remove backup files after verification
find . -name "*.py.bak" -delete
```

**Update shell scripts:**
```bash
find . -name "*.sh" -exec sed -i.bak "s|examples/|inputs/|g" {} \;
find . -name "*.sh.bak" -delete
```

**Update documentation:**
```bash
find . -name "*.md" -exec sed -i.bak "s|examples/|inputs/|g" {} \;
find . -name "*.md.bak" -delete
```

### 2. Filename Convention: Spaces Removed from Galaxy Names

**Reason:** Astronomy databases (NED, SIMBAD) use NGC3923, not NGC_3923. The underscore should only separate galaxy name from config name.

**Impact:** Scripts that parse output filenames must be updated.

#### Before (v1.x)
```
NGC_3923_clean.fits    (underscore in galaxy name)
IC_1459_noisy.fits
ESO_185-G054_clean.fits
```

#### After (v2.0)
```
NGC3923_clean.fits     (no underscore in galaxy name)
IC1459_noisy.fits
ESO185-G054_clean.fits
```

**Pattern:** `{galaxy_name}_{config_name}.fits`
- One underscore between galaxy and config
- Galaxy name has spaces removed (not replaced)
- Config name unchanged

#### Migration for Filename Parsing

**Python: Update regex patterns**

Before:
```python
# Old pattern (assumes underscore in galaxy name)
pattern = r"([A-Z]+_\d+)_(\w+)\.fits"
match = re.match(pattern, "NGC_3923_clean.fits")
galaxy = match.group(1)  # "NGC_3923"
config = match.group(2)  # "clean"
```

After:
```python
# New pattern (no underscore in galaxy name)
pattern = r"([A-Z]+\d+(?:-[A-Z]\d+)?)_(\w+)\.fits"
match = re.match(pattern, "NGC3923_clean.fits")
galaxy = match.group(1)  # "NGC3923"
config = match.group(2)  # "clean"
```

**Python: Update filename construction**

Before:
```python
galaxy_name = "NGC 3923"
config_name = "clean"
filename = f"{galaxy_name.replace(' ', '_')}_{config_name}.fits"
# Result: "NGC_3923_clean.fits"
```

After:
```python
from mockgal import sanitize_filename

galaxy_name = "NGC 3923"
config_name = "clean"
filename = f"{sanitize_filename(galaxy_name)}_{config_name}.fits"
# Result: "NGC3923_clean.fits"
```

**Shell: Update glob patterns**

Before:
```bash
# Old: assumes underscore separates all parts
for file in output/NGC_*_clean.fits; do
    # Process file
done
```

After:
```bash
# New: be more flexible with galaxy name format
for file in output/*_clean.fits; do
    # Extract galaxy name (everything before last underscore)
    galaxy=$(basename "$file" .fits | sed 's/_[^_]*$//')
    # Extract config (everything after last underscore)
    config=$(basename "$file" .fits | sed 's/^.*_//')
    echo "Galaxy: $galaxy, Config: $config"
done
```

#### Helper Functions for Migration

**Python: Flexible filename parser (works with old and new)**

```python
import re
from pathlib import Path

def parse_mockgal_filename(filepath):
    """
    Parse MockGal output filename (supports both v1.x and v2.0 formats).

    Parameters
    ----------
    filepath : str or Path
        Path to FITS file

    Returns
    -------
    tuple
        (galaxy_name, config_name)

    Examples
    --------
    >>> parse_mockgal_filename("NGC_3923_clean.fits")
    ('NGC_3923', 'clean')
    >>> parse_mockgal_filename("NGC3923_clean.fits")
    ('NGC3923', 'clean')
    """
    stem = Path(filepath).stem

    # Split on last underscore
    if '_' in stem:
        parts = stem.rsplit('_', 1)
        return parts[0], parts[1]
    else:
        # No underscore, assume entire name is galaxy
        return stem, None
```

**Python: Update existing filenames**

```python
from pathlib import Path
from mockgal import sanitize_filename

def migrate_output_filenames(output_dir):
    """
    Rename output files from v1.x to v2.0 convention.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing FITS files
    """
    output_dir = Path(output_dir)

    for old_path in output_dir.glob("*_*.fits"):
        # Parse old filename
        stem = old_path.stem
        parts = stem.rsplit('_', 1)

        if len(parts) == 2:
            galaxy_old, config = parts

            # Check if galaxy name has underscore (v1.x convention)
            if '_' in galaxy_old:
                # Convert: "NGC_3923" → "NGC3923"
                galaxy_new = galaxy_old.replace('_', '')

                # Construct new filename
                new_filename = f"{galaxy_new}_{config}.fits"
                new_path = old_path.parent / new_filename

                # Rename file
                print(f"Renaming: {old_path.name} → {new_filename}")
                old_path.rename(new_path)

# Usage
migrate_output_filenames("output/huang2013_test")
```

## Non-Breaking Changes

### Enhanced Visualization

New features available:
- PNG-only generation (without FITS)
- Custom colormaps
- Configurable contour levels
- High-DPI output

```python
from mockgal import visualize_galaxy

# Generate PNG without saving FITS
visualize_galaxy(image_array, metadata=meta, output_path="galaxy.png")

# Custom visualization
visualize_galaxy("galaxy.fits",
                cmap='magma',
                n_contours=15,
                sigma_smooth=2.0,
                dpi=300)
```

### Benchmark Improvements

Benchmarks now include:
- System information (CPU, memory, OS)
- Timestamp
- Human-readable markdown summary

```bash
python benchmarks/bench_engines.py
# Creates:
#   - benchmark_results.json (complete data)
#   - benchmark_results.md (summary report)
```

### Better Documentation

New README files in each directory:
- `tests/README.md` - Test suite coverage and usage
- `benchmarks/README.md` - Benchmark types and interpretation
- `inputs/README.md` - Input file descriptions and relationships
- `output/README.md` - Output organization and file formats

## Testing Your Migration

### 1. Update and Test Paths

```bash
# Test single galaxy mode
python mockgal.py \
    --models inputs/huang2013_models.yaml \
    --config inputs/huang2013_test_config.yaml \
    --galaxy "NGC 3923" \
    --workers 1 \
    -o output/migration_test/

# Verify output filename (should be NGC3923_clean.fits)
ls output/migration_test/
```

### 2. Run Test Suite

```bash
pytest tests/test_mockgal.py -v
```

All tests should pass, confirming:
- Path updates work correctly
- Filename sanitization works
- New visualization features work

### 3. Verify Your Scripts

Run your custom scripts with a small test case:
```bash
# Test with 2-3 galaxies before full batch
python your_script.py --test-mode
```

## Rollback Procedure

If you need to revert to v1.x:

```bash
# 1. Checkout previous version
git checkout v1.x

# 2. Folder structure reverts automatically (tracked by git)

# 3. Old output files (if needed)
# Reverse filename migration manually or regenerate
```

## Getting Help

If you encounter migration issues:

1. Check this guide first
2. Review the relevant README:
   - `tests/README.md` for test issues
   - `benchmarks/README.md` for benchmark issues
   - `inputs/README.md` for input file issues
   - `output/README.md` for output issues
3. Open an issue on GitHub with:
   - Error message
   - Command you ran
   - MockGal version (`python mockgal.py --version`)
   - System info (`python -c "import platform; print(platform.platform())"`)

## Summary Checklist

- [ ] Update all `examples/` → `inputs/` in scripts
- [ ] Update filename parsing logic (if applicable)
- [ ] Test with small dataset first
- [ ] Run test suite to verify
- [ ] Update any documentation or README files you maintain
- [ ] Regenerate any reference images if needed
- [ ] Update CI/CD pipelines (if applicable)
- [ ] Inform team members about changes
