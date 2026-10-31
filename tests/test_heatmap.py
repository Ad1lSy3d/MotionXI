"""
tests/test_heatmap.py
~~~~~~~~~~~~~~~~~~~~~
Unit tests for courtvision.analytics.heatmap.

Coverage
--------
1. Known positions map to expected grid cells.
2. Different players produce separate heatmaps.
3. Team heatmap aggregates valid player observations across tracks.
4. Invalid projections and uncalibrated observations are excluded.
5. Empty input and players with no valid observations handle gracefully.
6. Boundary and out-of-bounds coordinate handling.
7. JSON serialization and PNG file export.
8. Gaussian smoothing preserves total density and normalization invariants.
9. Metadata correctly identifies calibration scope and perspective limitations.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from courtvision.analytics.heatmap import (
    DEFAULT_BINS_X,
    DEFAULT_BINS_Y,
    DEFAULT_CALIBRATION_METADATA,
    HeatmapData,
    export_heatmap_json,
    export_heatmap_png,
    export_heatmaps_to_directory,
    generate_all_heatmaps,
    generate_player_heatmap,
    generate_team_heatmap,
    map_pitch_to_grid,
)
from courtvision.tracking.track_types import TrackedPlayer


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _create_obs(
    track_id: int,
    frame: int = 0,
    pitch_x: float = 10.0,
    pitch_y: float = 30.0,
    projection_valid: bool = True,
    in_calibrated_zone: bool = True,
) -> TrackedPlayer:
    """Helper to instantiate TrackedPlayer instances for heatmap tests."""
    return TrackedPlayer.create(
        frame=frame,
        timestamp=frame / 25.0,
        track_id=track_id,
        class_id=0,
        class_name="player",
        confidence=0.9,
        x1=0.0,
        y1=0.0,
        x2=50.0,
        y2=100.0,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        projection_valid=projection_valid,
        in_calibrated_zone=in_calibrated_zone,
    )


# ---------------------------------------------------------------------------
# Test 1: Known positions map to expected grid cells
# ---------------------------------------------------------------------------


def test_known_positions_map_to_expected_grid_cells():
    # 105 bins along 105m -> exactly 1.0m per column
    # 68 bins along 68m -> exactly 1.0m per row
    obs = [
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=20.0),
        _create_obs(track_id=1, pitch_x=5.0, pitch_y=30.0),
    ]

    heatmap = generate_player_heatmap(
        obs, track_id=1, bins_x=105, bins_y=68, sigma=None, normalize=False
    )

    assert heatmap.valid_samples == 2
    assert heatmap.grid.shape == (68, 105)

    # In grid[row, col], row corresponds to y and col corresponds to x
    assert heatmap.grid[20, 10] == 1.0
    assert heatmap.grid[30, 5] == 1.0

    # Ensure other cells are zero
    assert np.sum(heatmap.grid) == 2.0


def test_map_pitch_to_grid_proportional_scaling():
    # Test arbitrary grid resolution: 50 bins_x on 100m pitch, 20 bins_y on 50m pitch
    row, col = map_pitch_to_grid(
        pitch_x=25.0,
        pitch_y=12.5,
        bins_x=50,
        bins_y=20,
        pitch_length_m=100.0,
        pitch_width_m=50.0,
    )
    # 25 / 100 * 50 = 12.5 -> col 12
    # 12.5 / 50 * 20 = 5.0 -> row 5
    assert col == 12
    assert row == 5


# ---------------------------------------------------------------------------
# Test 2: Different players produce separate heatmaps
# ---------------------------------------------------------------------------


def test_different_players_produce_separate_heatmaps():
    obs = [
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=20.0),
        _create_obs(track_id=2, pitch_x=15.0, pitch_y=25.0),
    ]

    hm_p1 = generate_player_heatmap(obs, track_id=1, bins_x=105, bins_y=68, sigma=None)
    hm_p2 = generate_player_heatmap(obs, track_id=2, bins_x=105, bins_y=68, sigma=None)

    assert hm_p1.valid_samples == 1
    assert hm_p1.grid[20, 10] == 1.0
    assert hm_p1.grid[25, 15] == 0.0

    assert hm_p2.valid_samples == 1
    assert hm_p2.grid[25, 15] == 1.0
    assert hm_p2.grid[20, 10] == 0.0

    all_heatmaps = generate_all_heatmaps(obs, bins_x=105, bins_y=68, sigma=None)
    assert "player_1" in all_heatmaps
    assert "player_2" in all_heatmaps
    assert "team" in all_heatmaps


# ---------------------------------------------------------------------------
# Test 3: Team heatmap aggregates valid player observations
# ---------------------------------------------------------------------------


def test_team_heatmap_aggregates_valid_player_observations():
    obs = [
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=20.0),
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=20.0),  # same spot twice
        _create_obs(track_id=2, pitch_x=15.0, pitch_y=25.0),
        _create_obs(track_id=3, pitch_x=12.0, pitch_y=22.0),
    ]

    team_hm = generate_team_heatmap(obs, bins_x=105, bins_y=68, sigma=None)

    assert team_hm.entity_id == "team"
    assert team_hm.track_id is None
    assert team_hm.valid_samples == 4
    assert team_hm.grid[20, 10] == 2.0
    assert team_hm.grid[25, 15] == 1.0
    assert team_hm.grid[22, 12] == 1.0
    assert np.sum(team_hm.grid) == 4.0


# ---------------------------------------------------------------------------
# Test 4: Invalid and uncalibrated observations are excluded
# ---------------------------------------------------------------------------


def test_invalid_and_uncalibrated_observations_excluded():
    obs = [
        # Invalid projection flag
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=20.0, projection_valid=False),
        # Outside calibrated zone
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=20.0, in_calibrated_zone=False),
        # None coordinates
        _create_obs(track_id=1, pitch_x=None, pitch_y=20.0),  # type: ignore[arg-type]
        # NaN coordinates
        _create_obs(track_id=1, pitch_x=float("nan"), pitch_y=20.0),
        # Infinite coordinates
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=float("inf")),
    ]

    p_hm = generate_player_heatmap(obs, track_id=1, sigma=None)
    assert p_hm.valid_samples == 0
    assert p_hm.total_samples == 5
    assert np.sum(p_hm.grid) == 0.0

    team_hm = generate_team_heatmap(obs, sigma=None)
    assert team_hm.valid_samples == 0
    assert team_hm.total_samples == 5
    assert np.sum(team_hm.grid) == 0.0


# ---------------------------------------------------------------------------
# Test 5: Empty input and players with no valid observations
# ---------------------------------------------------------------------------


def test_empty_input_and_no_valid_observations():
    # Empty input
    p_hm_empty = generate_player_heatmap([], track_id=4)
    assert p_hm_empty.valid_samples == 0
    assert p_hm_empty.total_samples == 0
    assert np.sum(p_hm_empty.grid) == 0.0

    team_hm_empty = generate_team_heatmap([])
    assert team_hm_empty.valid_samples == 0
    assert team_hm_empty.total_samples == 0
    assert np.sum(team_hm_empty.grid) == 0.0

    all_hm_empty = generate_all_heatmaps([])
    assert "team" in all_hm_empty
    assert len(all_hm_empty) == 1

    # Request player not present in observations
    obs = [_create_obs(track_id=1, pitch_x=10.0, pitch_y=20.0)]
    p_hm_missing = generate_player_heatmap(obs, track_id=99)
    assert p_hm_missing.valid_samples == 0
    assert p_hm_missing.total_samples == 0
    assert np.sum(p_hm_missing.grid) == 0.0


# ---------------------------------------------------------------------------
# Test 6: Boundary and out-of-bounds handling
# ---------------------------------------------------------------------------


def test_boundary_and_out_of_bounds_handling():
    bins_x, bins_y = 105, 68
    obs = [
        # Exactly at origin (0, 0)
        _create_obs(track_id=1, pitch_x=0.0, pitch_y=0.0),
        # Exactly at pitch maximum (105, 68)
        _create_obs(track_id=1, pitch_x=105.0, pitch_y=68.0),
        # Out of bounds: negative x
        _create_obs(track_id=1, pitch_x=-2.0, pitch_y=20.0),
        # Out of bounds: excessive x
        _create_obs(track_id=1, pitch_x=110.0, pitch_y=20.0),
        # Out of bounds: negative y
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=-5.0),
        # Out of bounds: excessive y
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=75.0),
    ]

    hm = generate_player_heatmap(obs, track_id=1, bins_x=bins_x, bins_y=bins_y, sigma=None)

    assert hm.valid_samples == 2
    assert hm.out_of_bounds_samples == 4
    assert hm.grid[0, 0] == 1.0
    assert hm.grid[bins_y - 1, bins_x - 1] == 1.0

    # Ensure helper directly returns None for out of bounds
    assert map_pitch_to_grid(-0.1, 20.0, bins_x, bins_y) is None
    assert map_pitch_to_grid(105.1, 20.0, bins_x, bins_y) is None
    assert map_pitch_to_grid(20.0, -0.1, bins_x, bins_y) is None
    assert map_pitch_to_grid(20.0, 68.1, bins_x, bins_y) is None


# ---------------------------------------------------------------------------
# Test 7: JSON serialization and PNG file generation
# ---------------------------------------------------------------------------


def test_json_serialization_and_png_export(tmp_path: Path):
    obs = [
        _create_obs(track_id=7, pitch_x=10.0, pitch_y=20.0),
        _create_obs(track_id=7, pitch_x=12.0, pitch_y=25.0),
    ]
    hm = generate_player_heatmap(obs, track_id=7)

    # Test to_dict
    data_dict = hm.to_dict()
    assert data_dict["entity_id"] == "player_7"
    assert data_dict["track_id"] == 7
    assert data_dict["valid_samples"] == 2
    assert isinstance(data_dict["grid"], list)
    assert len(data_dict["grid"]) == DEFAULT_BINS_Y
    assert len(data_dict["grid"][0]) == DEFAULT_BINS_X

    # Export individual JSON
    json_path = tmp_path / "player_7.json"
    exported_json = export_heatmap_json(hm, json_path)
    assert exported_json.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["entity_id"] == "player_7"
    assert loaded["valid_samples"] == 2
    assert "metadata" in loaded
    assert loaded["metadata"]["calibration_scope"] == "penalty_box_region"

    # Export individual PNG
    png_path = tmp_path / "player_7.png"
    exported_png = export_heatmap_png(hm, png_path)
    assert exported_png.exists()
    assert exported_png.stat().st_size > 1000

    # Validate PNG magic bytes
    with open(png_path, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n"

    # Export batch directory
    all_hms = generate_all_heatmaps(obs)
    dir_path = tmp_path / "batch_heatmaps"
    manifest = export_heatmaps_to_directory(all_hms, dir_path)

    assert "team" in manifest
    assert "player_7" in manifest
    assert Path(manifest["team"]["png"]).exists()
    assert Path(manifest["team"]["json"]).exists()
    assert Path(manifest["player_7"]["png"]).exists()
    assert Path(manifest["player_7"]["json"]).exists()


# ---------------------------------------------------------------------------
# Test 8: Gaussian smoothing preserves density & normalization invariants
# ---------------------------------------------------------------------------


def test_gaussian_smoothing_preserves_density_and_normalization_invariants():
    obs = [
        _create_obs(track_id=1, pitch_x=10.0, pitch_y=20.0),
        _create_obs(track_id=1, pitch_x=12.0, pitch_y=25.0),
        _create_obs(track_id=1, pitch_x=14.0, pitch_y=30.0),
    ]

    # Without smoothing
    hm_raw = generate_player_heatmap(obs, track_id=1, sigma=None, normalize=False)
    raw_sum = float(np.sum(hm_raw.grid))
    assert raw_sum == 3.0

    # With smoothing (sigma=2.0)
    hm_smoothed = generate_player_heatmap(obs, track_id=1, sigma=2.0, normalize=False)
    smoothed_sum = float(np.sum(hm_smoothed.grid))
    # Gaussian smoothing with volume compensation MUST preserve total density
    assert smoothed_sum == pytest.approx(raw_sum, rel=1e-5)

    # With normalization
    hm_normalized = generate_player_heatmap(obs, track_id=1, sigma=1.5, normalize=True)
    norm_sum = float(np.sum(hm_normalized.grid))
    assert norm_sum == pytest.approx(1.0, rel=1e-5)

    # Empty observation smoothing does not error and stays 0
    hm_empty = generate_player_heatmap([], track_id=1, sigma=2.0, normalize=True)
    assert float(np.sum(hm_empty.grid)) == 0.0


# ---------------------------------------------------------------------------
# Test 9: Metadata contains calibration limitations
# ---------------------------------------------------------------------------


def test_metadata_contains_calibration_limitations():
    hm = generate_team_heatmap([])
    meta = hm.metadata

    assert meta["calibration_scope"] == "penalty_box_region"
    assert meta["full_pitch_calibration"] is False
    assert "warning" in meta
    assert "penalty box" in meta["warning"].lower()
    assert "calibrated_zone_m" in meta
    assert meta["calibrated_zone_m"]["x_range"] == [0.0, 16.5]
    assert meta["calibrated_zone_m"]["y_range"] == [13.84, 54.16]
    assert "coverage_limitation" in meta
