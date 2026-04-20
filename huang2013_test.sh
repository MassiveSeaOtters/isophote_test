#!/bin/bash

uv run python scripts/generate_mocks.py \
  --run-manifest inputs/huang2013/runs/huang2013_hsc_i_wide.yaml \
  --output ~/Dropbox/work/data/huang2013 \
  --galaxies "ESO 185-G054"

uv run python scripts/generate_mocks.py \
  --run-manifest inputs/huang2013/runs/huang2013_hsc_i_dud.yaml \
  --output ~/Dropbox/work/data/huang2013 \
  --galaxies "ESO 185-G054"
