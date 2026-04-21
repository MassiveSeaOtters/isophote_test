#!/bin/bash

uv run python scripts/generate_mocks.py \
  --run-manifest inputs/cs4g/runs/cs4g_hsc_i_wide.yaml \
  --output /Volumes/galaxy/isophote/s4g_mock

uv run python scripts/generate_mocks.py \
  --run-manifest inputs/cs4g/runs/cs4g_hsc_i_dud.yaml \
  --output /Volumes/galaxy/isophote/s4g_mock
