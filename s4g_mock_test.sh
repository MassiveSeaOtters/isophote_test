#!/bin/bash

uv run python scripts/generate_mocks.py \
  --run-manifest inputs/cs4g/runs/cs4g_hsc_i_wide.yaml \
  --output ~/Dropbox/work/data/s4g_mock \
  --galaxies NGC1433 NGC0275 NGC1357

uv run python scripts/generate_mocks.py \
  --run-manifest inputs/cs4g/runs/cs4g_hsc_i_dud.yaml \
  --output ~/Dropbox/work/data/s4g_mock \
  --galaxies NGC1433 NGC0275 NGC1357
