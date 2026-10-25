"""
courtvision.analytics.heatmap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Player and team occupancy heatmap generation and visualization for CourtVision AI.

Design & Coordinate System
--------------------------
* Pitch Dimensions: Standard FIFA 105.0 m (length, X axis) x 68.0 m (width, Y axis).
* Coordinate Mapping:
    - Pitch X in [0.0, pitch_length_m] maps horizontally to column index:
        col = clip(floor(x / pitch_length_m * bins_x), 0, bins_x - 1)
    - Pitch Y in [0.0, pitch_width_m] maps vertically to row index:
        row = clip(floor(y / pitch_width_m * bins_y), 0, bins_y - 1)
    - Grid array layout: shape (bins_y, bins_x) where index [row, col] corresponds
      to (pitch_y, pitch_x).
* Validity & Calibration Constraints:
    - Only observations where projection_valid is True, in_calibrated_zone is True,
      and pitch coordinates are finite are included.
    - Points outside [0.0, pitch_length_m] x [0.0, pitch_width_m] are rejected as
      out-of-bounds.
    - Calibration Warning: Homography is calibrated solely from the 18-yard penalty box.
      Uncalibrated pitch areas do not have reliable tracking coverage under the
      current single-box homography.
* Output:
    - JSON density data with grid matrices and calibration metadata.
    - Publication-grade PNG pitch heatmaps with pitch boundary markings and
      calibration zone annotations.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from courtvision.tracking.track_types import TrackedPlayer

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_PITCH_LENGTH_M: float = 105.0
DEFAULT_PITCH_WIDTH_M: float = 68.0
DEFAULT_BINS_X: int = 105
DEFAULT_BINS_Y: int = 68
DEFAULT_SIGMA: float = 1.5

DEFAULT_CALIBRATION_METADATA: Dict[str, Any] = {
    "calibration_scope": "penalty_box_region",
    "full_pitch_calibration": False,
    "calibrated_zone_m": {
        "x_range": [0.0, 16.5],
        "y_range": [13.84, 54.16],
    },
    "warning": (
        "H is calibrated solely from 4 points of the 18-yard penalty box. "
        "Points outside the penalty-box region suffer from perspective divergence "
        "and must not be treated as trustworthy full-pitch coordinates."
    ),
    "coverage_limitation": (
        "Current occupancy heatmaps reflect player presence strictly within or near "
        "the calibrated penalty-box area. Uncalibrated pitch areas do not have reliable "
        "tracking coverage under the current single-box homography."
    ),
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class HeatmapData:
    """Container for pitch occupancy heatmap density and metadata.

    Attributes
    ----------
    entity_id:
        Identifier for the heatmap source (e.g. 'team', 'player_4').
    track_id:
        Integer track ID if player-specific, or None for team-level aggregation.
    bins_x:
        Number of grid bins along the pitch length (X axis).
    bins_y:
        Number of grid bins along the pitch width (Y axis).
    pitch_length_m:
        Pitch length in meters (default 105.0).
    pitch_width_m:
        Pitch width in meters (default 68.0).
    total_samples:
        Total observations considered for this entity.
    valid_samples:
        Count of valid observations accumulated into the grid.
    out_of_bounds_samples:
        Count of observations rejected because pitch coordinates fell outside
        [0, pitch_length_m] x [0, pitch_width_m].
    grid:
        2D numpy array of shape (bins_y, bins_x) representing occupancy density.
    metadata:
        Calibration metadata and limitation disclaimers.
    """

    entity_id: str
    track_id: Optional[int]
    bins_x: int
    bins_y: int
    pitch_length_m: float
    pitch_width_m: float
    total_samples: int
    valid_samples: int
    out_of_bounds_samples: int
    grid: np.ndarray
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert heatmap data to a JSON-serializable dictionary."""
        return {
            "entity_id": self.entity_id,
            "track_id": self.track_id,
            "bins_x": self.bins_x,
            "bins_y": self.bins_y,
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "out_of_bounds_samples": self.out_of_bounds_samples,
            "grid": self.grid.tolist(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Coordinate Mapping & Validation Helpers
# ---------------------------------------------------------------------------


def _is_usable_observation(obs: TrackedPlayer) -> bool:
    """Check if observation carries a finite projected position within the calibrated zone."""
    return bool(
        obs.projection_valid
        and obs.in_calibrated_zone
        and obs.pitch_x is not None
        and obs.pitch_y is not None
        and math.isfinite(obs.pitch_x)
        and math.isfinite(obs.pitch_y)
    )


def map_pitch_to_grid(
    pitch_x: float,
    pitch_y: float,
    bins_x: int,
    bins_y: int,
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
) -> Optional[Tuple[int, int]]:
    """Map continuous pitch coordinates (meters) to 2D grid indices (row, col).

    Returns
    -------
    (row, col) tuple where row in [0, bins_y - 1] corresponds to pitch_y,
    and col in [0, bins_x - 1] corresponds to pitch_x.
    Returns None if coordinates fall outside the pitch boundaries.
    """
    if pitch_x < 0.0 or pitch_x > pitch_length_m:
        return None
    if pitch_y < 0.0 or pitch_y > pitch_width_m:
        return None

    # Handle boundary edge case exactly at pitch_length / pitch_width
    col = int(pitch_x / pitch_length_m * bins_x)
    row = int(pitch_y / pitch_width_m * bins_y)

    col = min(col, bins_x - 1)
    row = min(row, bins_y - 1)

    return row, col


def _apply_smoothing_and_normalization(
    grid: np.ndarray,
    sigma: Optional[float],
    normalize: bool,
) -> np.ndarray:
    """Apply optional Gaussian smoothing and normalization while preserving invariants."""
    result = grid.astype(np.float64, copy=True)
    raw_sum = float(np.sum(result))

    if sigma is not None and sigma > 0.0 and raw_sum > 0.0:
        smoothed = gaussian_filter(result, sigma=sigma, mode="constant", cval=0.0)
        smoothed_sum = float(np.sum(smoothed))
        if smoothed_sum > 0.0:
            # Preserves total density sum invariant despite boundary cutoff
            result = smoothed * (raw_sum / smoothed_sum)

    if normalize and raw_sum > 0.0:
        total = float(np.sum(result))
        if total > 0.0:
            result = result / total

    return result


# ---------------------------------------------------------------------------
# Core Generation Functions
# ---------------------------------------------------------------------------


def generate_player_heatmap(
    observations: Sequence[TrackedPlayer],
    track_id: int,
    bins_x: int = DEFAULT_BINS_X,
    bins_y: int = DEFAULT_BINS_Y,
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
    sigma: Optional[float] = DEFAULT_SIGMA,
    normalize: bool = False,
) -> HeatmapData:
    """Generate an occupancy heatmap for an individual tracked player.

    Parameters
    ----------
    observations:
        Sequence of TrackedPlayer instances.
    track_id:
        Target player track ID.
    bins_x:
        Grid resolution along pitch length.
    bins_y:
        Grid resolution along pitch width.
    pitch_length_m:
        Pitch length in meters.
    pitch_width_m:
        Pitch width in meters.
    sigma:
        Standard deviation for Gaussian kernel. If None or 0, smoothing is skipped.
    normalize:
        If True, normalizes total grid sum to 1.0 (probability distribution).

    Returns
    -------
    HeatmapData
    """
    raw_grid = np.zeros((bins_y, bins_x), dtype=np.float64)
    total_samples = 0
    valid_samples = 0
    out_of_bounds = 0

    for obs in observations:
        if obs.track_id != track_id:
            continue
        total_samples += 1

        if not _is_usable_observation(obs):
            continue

        indices = map_pitch_to_grid(
            float(obs.pitch_x),  # type: ignore[arg-type]
            float(obs.pitch_y),  # type: ignore[arg-type]
            bins_x=bins_x,
            bins_y=bins_y,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )

        if indices is None:
            out_of_bounds += 1
            continue

        row, col = indices
        raw_grid[row, col] += 1.0
        valid_samples += 1

    processed_grid = _apply_smoothing_and_normalization(
        raw_grid, sigma=sigma, normalize=normalize
    )

    metadata = dict(DEFAULT_CALIBRATION_METADATA)
    metadata["smoothing_sigma"] = sigma
    metadata["normalized"] = normalize

    return HeatmapData(
        entity_id=f"player_{track_id}",
        track_id=track_id,
        bins_x=bins_x,
        bins_y=bins_y,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        total_samples=total_samples,
        valid_samples=valid_samples,
        out_of_bounds_samples=out_of_bounds,
        grid=processed_grid,
        metadata=metadata,
    )


def generate_team_heatmap(
    observations: Sequence[TrackedPlayer],
    bins_x: int = DEFAULT_BINS_X,
    bins_y: int = DEFAULT_BINS_Y,
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
    sigma: Optional[float] = DEFAULT_SIGMA,
    normalize: bool = False,
) -> HeatmapData:
    """Generate an aggregate team occupancy heatmap from all valid tracked players.

    Parameters
    ----------
    observations:
        Sequence of TrackedPlayer instances across all active players.
    bins_x:
        Grid resolution along pitch length.
    bins_y:
        Grid resolution along pitch width.
    pitch_length_m:
        Pitch length in meters.
    pitch_width_m:
        Pitch width in meters.
    sigma:
        Standard deviation for Gaussian kernel. If None or 0, smoothing is skipped.
    normalize:
        If True, normalizes total grid sum to 1.0.

    Returns
    -------
    HeatmapData with entity_id='team' and track_id=None.
    """
    raw_grid = np.zeros((bins_y, bins_x), dtype=np.float64)
    total_samples = len(observations)
    valid_samples = 0
    out_of_bounds = 0

    for obs in observations:
        if not _is_usable_observation(obs):
            continue

        indices = map_pitch_to_grid(
            float(obs.pitch_x),  # type: ignore[arg-type]
            float(obs.pitch_y),  # type: ignore[arg-type]
            bins_x=bins_x,
            bins_y=bins_y,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )

        if indices is None:
            out_of_bounds += 1
            continue

        row, col = indices
        raw_grid[row, col] += 1.0
        valid_samples += 1

    processed_grid = _apply_smoothing_and_normalization(
        raw_grid, sigma=sigma, normalize=normalize
    )

    metadata = dict(DEFAULT_CALIBRATION_METADATA)
    metadata["smoothing_sigma"] = sigma
    metadata["normalized"] = normalize

    return HeatmapData(
        entity_id="team",
        track_id=None,
        bins_x=bins_x,
        bins_y=bins_y,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        total_samples=total_samples,
        valid_samples=valid_samples,
        out_of_bounds_samples=out_of_bounds,
        grid=processed_grid,
        metadata=metadata,
    )


def generate_all_heatmaps(
    observations: Sequence[TrackedPlayer],
    bins_x: int = DEFAULT_BINS_X,
    bins_y: int = DEFAULT_BINS_Y,
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
    sigma: Optional[float] = DEFAULT_SIGMA,
    normalize: bool = False,
) -> Dict[str, HeatmapData]:
    """Generate both the team aggregate and all individual player heatmaps.

    Returns
    -------
    Dictionary mapping entity keys ('team', 'player_{id}') to HeatmapData instances.
    """
    results: Dict[str, HeatmapData] = {}

    # Team aggregate
    results["team"] = generate_team_heatmap(
        observations=observations,
        bins_x=bins_x,
        bins_y=bins_y,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        sigma=sigma,
        normalize=normalize,
    )

    # Distinct player IDs
    track_ids = sorted({obs.track_id for obs in observations})
    for tid in track_ids:
        key = f"player_{tid}"
        results[key] = generate_player_heatmap(
            observations=observations,
            track_id=tid,
            bins_x=bins_x,
            bins_y=bins_y,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
            sigma=sigma,
            normalize=normalize,
        )

    return results


# ---------------------------------------------------------------------------
# Export & Visualization Functions
# ---------------------------------------------------------------------------


def _draw_pitch_markings(ax: plt.Axes, pitch_length: float, pitch_width: float) -> None:
    """Draw tactical football pitch markings on matplotlib axes."""
    line_color = "white"
    lw = 1.5

    # Boundary touchlines and goal lines
    pitch_rect = patches.Rectangle(
        (0, 0), pitch_length, pitch_width, linewidth=lw, edgecolor=line_color, facecolor="none"
    )
    ax.add_patch(pitch_rect)

    # Halfway line
    ax.plot([pitch_length / 2.0, pitch_length / 2.0], [0, pitch_width], color=line_color, linewidth=lw)

    # Center circle
    center_circle = patches.Circle(
        (pitch_length / 2.0, pitch_width / 2.0), 9.15, linewidth=lw, edgecolor=line_color, facecolor="none"
    )
    ax.add_patch(center_circle)

    # Center spot
    center_spot = patches.Circle((pitch_length / 2.0, pitch_width / 2.0), 0.5, facecolor=line_color)
    ax.add_patch(center_spot)

    # Left penalty area (16.5m depth, 40.32m width centered)
    box_depth = 16.5
    box_y_min = 13.84
    box_height = 40.32
    left_penalty = patches.Rectangle(
        (0, box_y_min), box_depth, box_height, linewidth=lw, edgecolor=line_color, facecolor="none"
    )
    ax.add_patch(left_penalty)

    # Left 6-yard box (5.5m depth, 18.32m width centered)
    left_six = patches.Rectangle(
        (0, 24.84), 5.5, 18.32, linewidth=lw, edgecolor=line_color, facecolor="none"
    )
    ax.add_patch(left_six)

    # Right penalty area
    right_penalty = patches.Rectangle(
        (pitch_length - box_depth, box_y_min),
        box_depth,
        box_height,
        linewidth=lw,
        edgecolor=line_color,
        facecolor="none",
    )
    ax.add_patch(right_penalty)

    # Right 6-yard box
    right_six = patches.Rectangle(
        (pitch_length - 5.5, 24.84), 5.5, 18.32, linewidth=lw, edgecolor=line_color, facecolor="none"
    )
    ax.add_patch(right_six)

    # Annotation highlighting calibrated 18-yard box region
    calib_box = patches.Rectangle(
        (0, box_y_min),
        box_depth,
        box_height,
        linewidth=2.0,
        edgecolor="#FFD700",
        linestyle="--",
        facecolor="none",
        label="Calibrated 18-yd Box",
    )
    ax.add_patch(calib_box)


def export_heatmap_png(
    heatmap: HeatmapData,
    output_path: Union[str, Path],
    colormap: str = "hot",
    title: Optional[str] = None,
    dpi: int = 150,
) -> Path:
    """Export a heatmap as a high-resolution PNG image overlaying a pitch graphic.

    Parameters
    ----------
    heatmap:
        HeatmapData instance.
    output_path:
        Target file path for the exported image.
    colormap:
        Matplotlib colormap name (e.g. 'hot', 'viridis', 'magma').
    title:
        Plot title. Defaults to entity identification.
    dpi:
        Image resolution in dots per inch.

    Returns
    -------
    Path to the written PNG file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor="#0B1A0E")
    ax.set_facecolor("#1B4D24")

    # Draw density grid
    # Extent maps array indices to pitch meters: [x_min, x_max, y_min, y_max]
    extent = [0.0, heatmap.pitch_length_m, 0.0, heatmap.pitch_width_m]

    # Mask zero or negligible values for visual clarity if non-empty
    grid_display = heatmap.grid.copy()
    max_val = np.max(grid_display)
    if max_val > 0.0:
        im = ax.imshow(
            grid_display,
            origin="lower",
            extent=extent,
            cmap=colormap,
            alpha=0.75,
            interpolation="gaussian",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        cbar.set_label("Occupancy Density", color="white")
    else:
        # Display empty field if no samples
        ax.imshow(
            grid_display,
            origin="lower",
            extent=extent,
            cmap=colormap,
            alpha=0.2,
        )

    # Overlay pitch lines
    _draw_pitch_markings(ax, heatmap.pitch_length_m, heatmap.pitch_width_m)

    ax.set_xlim(-2.0, heatmap.pitch_length_m + 2.0)
    ax.set_ylim(-2.0, heatmap.pitch_width_m + 2.0)
    ax.set_aspect("equal")

    # Labels and titles
    display_title = title or f"Occupancy Heatmap: {heatmap.entity_id}"
    ax.set_title(display_title, color="white", fontsize=14, fontweight="bold", pad=12)

    subtitle = (
        f"Valid Samples: {heatmap.valid_samples} | "
        f"Calibrated Scope: 18-yard box [0-16.5m, 13.8-54.2m] (Yellow Dashed)"
    )
    ax.set_xlabel(f"Pitch Length (m)\n{subtitle}", color="white", fontsize=10, labelpad=8)
    ax.set_ylabel("Pitch Width (m)", color="white", fontsize=10, labelpad=8)
    ax.tick_params(colors="white")

    plt.tight_layout()
    fig.savefig(str(path), dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    return path


def export_heatmap_json(
    heatmap: HeatmapData,
    output_path: Union[str, Path],
) -> Path:
    """Export heatmap grid matrix and metadata as a JSON file.

    Parameters
    ----------
    heatmap:
        HeatmapData instance.
    output_path:
        Target file path for JSON output.

    Returns
    -------
    Path to the written JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(heatmap.to_dict(), f, indent=2)

    return path


def export_heatmaps_to_directory(
    heatmaps: Dict[str, HeatmapData],
    output_dir: Union[str, Path],
    colormap: str = "hot",
    dpi: int = 150,
) -> Dict[str, Dict[str, str]]:
    """Export all heatmaps in a dictionary as both PNG images and JSON files to a directory.

    Parameters
    ----------
    heatmaps:
        Dictionary mapping entity keys to HeatmapData instances.
    output_dir:
        Directory path where files will be created.
    colormap:
        Matplotlib colormap name.
    dpi:
        Resolution for PNG images.

    Returns
    -------
    Dictionary mapping each entity key to paths: {'png': str, 'json': str}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, str]] = {}

    for key, hm in heatmaps.items():
        png_filename = f"{key}_heatmap.png"
        json_filename = f"{key}_heatmap.json"

        png_path = export_heatmap_png(hm, out / png_filename, colormap=colormap, dpi=dpi)
        json_path = export_heatmap_json(hm, out / json_filename)

        manifest[key] = {
            "png": str(png_path),
            "json": str(json_path),
        }

    return manifest
