from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "inputs"
    / "huang2013"
    / "scripts"
    / "generate_huang2013_mocks.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "generate_huang2013_mocks_module",
    MODULE_PATH,
)
generate_huang2013_mocks = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = generate_huang2013_mocks
MODULE_SPEC.loader.exec_module(generate_huang2013_mocks)


def write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "run_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def test_resolve_run_manifest_applies_defaults_and_row_overrides(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        {
            "run_name": "test_run",
            "model_file": "inputs/huang2013/models/huang2013_models.yaml",
            "defaults": {
                "pixel_scale": 0.2,
                "noise_enabled": False,
                "sky_sb_limit": None,
            },
            "configs": [
                {"name": "z005_clean", "redshift": 0.05, "size_factor": 6.0},
                {
                    "name": "z020_noisy",
                    "redshift": 0.2,
                    "size_factor": 8.0,
                    "noise_enabled": True,
                    "sky_sb_limit": 24.5,
                },
            ],
        },
    )

    manifest = generate_huang2013_mocks.resolve_run_manifest(manifest_path)

    assert manifest.run_name == "test_run"
    assert manifest.rows[0].config_values["pixel_scale"] == pytest.approx(0.2)
    assert manifest.rows[0].config_values["size_factor"] == pytest.approx(6.0)
    assert manifest.rows[0].config_values["noise_enabled"] is False
    assert manifest.rows[1].config_values["noise_enabled"] is True
    assert manifest.rows[1].config_values["size_factor"] == pytest.approx(8.0)
    assert manifest.rows[1].config_values["sky_sb_limit"] == pytest.approx(24.5)


def test_resolve_run_manifest_rejects_unknown_row_keys(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        {
            "run_name": "bad_keys",
            "model_file": "inputs/huang2013/models/huang2013_models.yaml",
            "configs": [
                {"name": "z005_clean", "redshift": 0.05, "size_factor": 6.0, "unknown_option": 3},
            ],
        },
    )

    with pytest.raises(ValueError, match="unknown keys"):
        generate_huang2013_mocks.resolve_run_manifest(manifest_path)


def test_resolve_run_manifest_rejects_duplicate_config_names(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        {
            "run_name": "dup_names",
            "model_file": "inputs/huang2013/models/huang2013_models.yaml",
            "configs": [
                {"name": "same_name", "redshift": 0.05, "size_factor": 6.0},
                {"name": "same_name", "redshift": 0.2, "size_factor": 6.0},
            ],
        },
    )

    with pytest.raises(ValueError, match="Duplicate config name"):
        generate_huang2013_mocks.resolve_run_manifest(manifest_path)


def test_resolve_run_manifest_requires_noise_driver_when_noise_enabled(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        {
            "run_name": "bad_noise",
            "model_file": "inputs/huang2013/models/huang2013_models.yaml",
            "configs": [
                {"name": "broken_noise", "redshift": 0.05, "size_factor": 6.0, "noise_enabled": True},
            ],
        },
    )

    with pytest.raises(ValueError, match="does not provide any noise driver"):
        generate_huang2013_mocks.resolve_run_manifest(manifest_path)


def test_select_rows_filters_by_requested_names(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        {
            "run_name": "selection",
            "model_file": "inputs/huang2013/models/huang2013_models.yaml",
            "configs": [
                {"name": "z005_clean", "redshift": 0.05, "size_factor": 6.0},
                {"name": "z020_noise", "redshift": 0.2, "size_factor": 6.0, "noise_enabled": True, "sky_sb_limit": 24.5},
            ],
        },
    )
    manifest = generate_huang2013_mocks.resolve_run_manifest(manifest_path)

    rows = generate_huang2013_mocks.select_rows(manifest.rows, ["z020_noise"])

    assert [row.name for row in rows] == ["z020_noise"]


def test_write_resolved_manifest_and_metadata_record_selected_rows(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        {
            "run_name": "artifacts",
            "model_file": "inputs/huang2013/models/huang2013_models.yaml",
            "configs": [
                {"name": "z005_clean", "redshift": 0.05, "size_factor": 6.0},
                {"name": "z050_noise", "redshift": 0.5, "size_factor": 6.0, "noise_enabled": True, "sky_sb_limit": 24.5},
            ],
        },
    )
    manifest = generate_huang2013_mocks.resolve_run_manifest(manifest_path)
    selected_rows = generate_huang2013_mocks.select_rows(manifest.rows, ["z050_noise"])
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    resolved_manifest_path = generate_huang2013_mocks.write_resolved_manifest(
        output_dir,
        manifest,
        selected_rows,
    )
    metadata_path = generate_huang2013_mocks.write_run_metadata(
        output_base=output_dir,
        manifest=manifest,
        rows=selected_rows,
        galaxies=[],
        selected_galaxy_names=None,
        selected_config_names=["z050_noise"],
        warnings=[],
    )

    resolved_payload = yaml.safe_load(resolved_manifest_path.read_text(encoding="utf-8"))
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert [row["name"] for row in resolved_payload["configs"]] == ["z050_noise"]
    assert metadata_payload["config_count"] == 1
    assert metadata_payload["selected_config_names"] == ["z050_noise"]
    assert metadata_payload["resolved_rows"][0]["name"] == "z050_noise"
    assert metadata_payload["resolved_rows"][0]["size_factor"] == pytest.approx(6.0)


def test_resolve_run_manifest_requires_row_level_size_factor(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        {
            "run_name": "missing_size_factor",
            "model_file": "inputs/huang2013/models/huang2013_models.yaml",
            "defaults": {
                "size_factor": 6.0,
            },
            "configs": [
                {"name": "z005_clean", "redshift": 0.05},
            ],
        },
    )

    with pytest.raises(ValueError, match="missing required keys: size_factor"):
        generate_huang2013_mocks.resolve_run_manifest(manifest_path)


def test_build_image_config_uses_descriptive_name() -> None:
    config = generate_huang2013_mocks.build_image_config(
        "z020_baseline_noise",
        {
            **generate_huang2013_mocks.SCRIPT_DEFAULTS,
            "noise_enabled": True,
            "sky_sb_limit": 24.5,
        },
    )

    assert config.name == "z020_baseline_noise"
    assert config.sky_sb_limit == pytest.approx(24.5)
