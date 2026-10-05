# CourtVision AI (AthletaTrack)

> Physics-informed multi-object tracking for broadcast football that resolves identity switching in dense scrums, paired with an automated FotMob-style physical and spatial analytics engine.

---

## 1. Overview

In broadcast football video, conventional multi-object tracking algorithms (ByteTrack, DeepSORT, Kalman filters) fail during high-density contact events—such as set-piece walls, corner kick scrums, and sliding tackles. When players overlap ($\text{IoU} > 0.35$), visual features blend, bounding boxes merge, and linear motion assumptions collapse, causing severe **identity switches (IDSw)** and **track fragmentation**. **CourtVision AI** addresses this challenge by introducing a physics-grounded **Kinematic Memory Tracker (KMT)**. By monitoring bounding-box collision graphs, KMT caches pre-collision momentum vectors, extrapolates constant-momentum "ghost trajectories" through occlusions, and re-identifies players via physics-gated Hungarian association upon separation. The resulting coordinate stream is projected onto FIFA pitch coordinates ($105\text{m} \times 68\text{m}$) via homography to generate broadcast-grade physical analytics: distance covered, top speed, sprint counts ($>25\text{ km/h}$), 2D Gaussian KDE heatmaps, and dynamic match ratings.

---

## 2. Architecture Pipeline

```text
[Raw Broadcast Match Video (1080p @ 25/30 FPS)]
                    │
                    ▼
     [YOLOv11m Player Detection Engine] ──────────► Players, Referees, Goalkeepers, Ball
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[4-Point Pitch Homography]  [HSV Kit Clustering]
(Pixel u,v ──► Pitch X,Y)   (Upper-body team IDs)
        │                       │
        └───────────┬───────────┘
                    ▼
          [Base Tracker: ByteTrack]
                    │
         [IoU >= 0.35 Scrum Cluster?]
         ├── No  ──► Standard Linear Bipartite Matching
         └── Yes ──► [Kinematic Memory Module (KMT - Milestone 4)]
                     ├── Cache pre-collision momentum (Vx, Vy, Ax, Ay)
                     ├── Mute corrupted visual/box embeddings
                     ├── Extrapolate kinematic ghost trajectories
                     └── Physics-gated Hungarian matching upon emergence
                                 │
                                 ▼
           [Continuous Trajectory Stream: (ID, Frame, X_m, Y_m)]
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
[Physical Engine]       [Spatial Heatmaps]      [Match Rating Engine]
• Total Distance (km)   • 2D Gaussian KDE       • Base 6.0 + impact score
• Sprints (>25 km/h)    • Positional drift      • High-intensity outputs
• Top Speed (km/h)      • Zone occupancy        • Possession contribution
                                 │
                                 ▼
                       [FastAPI REST Engine] ──► Broadcast Overlays / Webhooks / JSON
```

---

## 3. Current Project Status

- **Milestone 1 (Complete):** YOLOv11m fine-tuned player detection pipeline verified on Apple Silicon MPS hardware with automatic CPU fallback. Achieved 93.1% mAP@50, 95.4% precision, and 91.6% recall.
- **Milestone 2 (Complete):** Modular ByteTrack baseline implemented with ground-contact foot projection, 15-frame history trails, and collision telemetry.
- **Milestone 3A (Complete):** Acquired official **SoccerNet Tracking 2023** evaluation benchmark (`tracking-2023` test split), established 1-to-1 video/GT frame mapping, evaluated the ByteTrack baseline with official **TrackEval** tooling, and generated empirical failure analysis.
- **Milestone 4 (Upcoming):** Design and implementation of the **Kinematic Memory Tracker (KMT)** to resolve dense scrum identity switches.

---

## 4. Requirements & Hardware Support

- **Python:** `>=3.12` (tested with Python 3.12.9 / 3.12.13).
- **Environment & Package Manager:** [Astral UV](https://github.com/astral-sh/uv) (version `>=0.6.0`).
- **Hardware Acceleration:**
  - **Apple Silicon (M1/M2/M3/M4):** Automatic Metal Performance Shaders (`mps`) acceleration.
  - **NVIDIA GPU:** Automatic `cuda` acceleration when available.
  - **CPU:** Fully supported fallback; pipelines automatically default to `cpu` if hardware acceleration is unavailable.

---

## 5. UV Environment Setup & Installation

This project strictly uses **UV** for deterministic environment management. **Do NOT run `pip install`, `python -m venv`, `conda`, or `poetry`.**

### Clone & Sync

```bash
# 1. Clone repository
git clone <repository-url>
cd MotionXI

# 2. Sync environment and build editable courtvision package
uv sync
```

`uv sync` automatically installs all runtime and development dependencies into a local `.venv/` and registers the `courtvision` package. No manual virtual environment creation or activation is required.

---

## 6. Running Automated Tests

Run the complete test suite (32 unit and regression tests) through UV:

```bash
uv run pytest -v
```

The test suite covers:
- YOLOv11m detector loading and foot-contact calculation
- 4-point homography boundary validation and point projection
- ByteTrack track ID persistence and state serialization
- Video metadata reading and container vs. stream frame validation
- MOT ground truth parsing and video-to-GT frame alignment
- Hungarian bipartite matching and crowding classification
- Synthetic identity swap, track fragmentation, and occlusion recovery metrics
- Package imports and graceful error handling for missing models/videos

---

## 7. Model Checkpoint Instructions

The trained player detection checkpoint is:

```text
weights/best.pt  (~39 MB, YOLOv11m fine-tuned on SoccerNet tracking frames)
```

### Checkpoint Acquisition Strategy

To keep the git repository lightweight and avoid bandwidth quota limits, model binaries (`weights/*.pt`) are excluded from Git history via `.gitignore`.

1. Obtain `best.pt` from the team storage or model release artifact repository.
2. Place the file into the `weights/` directory:
   ```bash
   cp /path/to/downloaded/best.pt weights/best.pt
   ```
3. If `weights/best.pt` is missing, detector scripts will fail gracefully with an informative error message:
   ```text
   FileNotFoundError: Model checkpoint not found at: weights/best.pt.
   Please acquire the trained YOLOv11m checkpoint ('best.pt') and place it in the weights/ directory.
   ```

---

## 8. Dataset Instructions & Benchmark Setup

### Demo Video (External Highlight)

- Location: `data/test_match.mp4` (1080p @ 30 FPS, 600 frames, ~20.0s).
- Used for qualitative visual verification, detection bounding box overlays, and homography visualization.

### Official SoccerNet Tracking 2023 Evaluation Benchmark

- Task: `tracking-2023` from `SoccerNet/SN-Tracking-2023` (Hugging Face).
- Selected Evaluation Subset (held-out test split, 3,750 frames total @ 25 FPS):
  - `SNMOT-116`: Corner kick scrum (heavy box congestion, 27 tracklets, 750 frames)
  - `SNMOT-117`: Offside line crossing (overlapping lateral runs, 31 tracklets, 750 frames)
  - `SNMOT-118`: Goalmouth scramble & shot off target (25 tracklets, 750 frames)
  - `SNMOT-119`: Defensive clearance scrum (transitional density, 26 tracklets, 750 frames)
  - `SNMOT-120`: Sliding tackle contact (rapid player convergence, 27 tracklets, 750 frames)
- Expected Directory Layout:
  ```text
  data/
  ├── homography_matrix.json
  ├── test_match.mp4
  └── soccernet_tracking/
      ├── annotations/
      │   ├── SNMOT-116/gt/gt.txt
      │   ├── ...
      │   └── SNMOT-120/gt/gt.txt
      └── videos/
          ├── SNMOT-116.mp4
          ├── ...
          └── SNMOT-120.mp4
  ```
- **Note:** Benchmark videos and raw frames are excluded from Git (`data/soccernet_tracking/` in `.gitignore`). Ground truth annotations and evaluation videos are downloaded selectively via the official `SoccerNet` package.

---

## 9. Command Documentation

All commands are executed using `uv run`:

### A. Run Player Detection Pipeline (Milestone 1)

```bash
uv run python scripts/run_detection.py \
    --video data/test_match.mp4 \
    --weights weights/best.pt \
    --homography data/homography_matrix.json \
    --output-dir outputs/detection
```

Outputs generated:
- `outputs/detection/annotated.mp4`: Video with bounding boxes, confidence badges, foot markers, and telemetry HUD.
- `outputs/detection/detections.csv`: Frame-by-frame tabular detection logs.
- `outputs/detection/detections.json`: Structured detection records.
- `outputs/detection/run_metadata.json`: Runtime, hardware, and throughput telemetry.

### B. Run ByteTrack Baseline Pipeline (Milestone 2)

```bash
uv run python scripts/run_tracking.py \
    --video data/test_match.mp4 \
    --weights weights/best.pt \
    --homography data/homography_matrix.json \
    --output-dir outputs/tracking
```

Outputs generated:
- `outputs/tracking/bytetrack_annotated.mp4`: Video overlay with persistent IDs and 15-frame history trails.
- `outputs/tracking/tracks.csv`: Complete tracking log.
- `outputs/tracking/track_summary.csv`: Per-track metrics (length, first/last frame, confidence).
- `outputs/tracking/crowded_frames.csv`: Crowding telemetry sorted by severity.
- `outputs/tracking/id_switch_candidates.csv`: Heuristic identity switch indicators.
- `outputs/tracking/tracking_metadata.json`: Tracking configuration and hardware metrics.

### C. Run Official Tracking Evaluation (Milestone 3A)

```bash
uv run python scripts/evaluate_tracking.py \
    --ground-truth data/soccernet_tracking/annotations \
    --videos-dir data/soccernet_tracking/videos \
    --weights weights/best.pt \
    --output-dir outputs/evaluation
```

Outputs generated:
- `outputs/evaluation/metrics.csv` & `metrics.json`: Standard TrackEval metrics across subsets.
- `outputs/evaluation/id_switch_events.csv`: 320 confirmed ID switches with root-cause categorization.
- `outputs/evaluation/identity_fragmentation.csv`: Per-player track fragmentation metrics.
- `outputs/evaluation/occlusion_recovery.csv`: Occlusion duration and recovery audit.
- `outputs/evaluation/diagnostics/`: Visual overlays illustrating confirmed failure cases.

---

## 10. Quantitative Baseline Metrics (TrackEval)

Benchmark performance of the unmodified YOLOv11m + ByteTrack pipeline on the SoccerNet Tracking 2023 evaluation subset:

| Metric | Overall | Normal Subset | Crowded/Occluded Subset |
|---|---:|---:|---:|
| **HOTA** | **72.81%** | 79.67% | 72.62% |
| **AssA (Association Accuracy)** | **61.48%** | **72.84%** | **61.71%** |
| **MOTA** | **83.26%** | 84.01% | 82.33% |
| **IDSW (Identity Switches)** | **294** | 144 | **231** |
| **Frag (Track Fragmentations)** | **649** | 245 | **452** |
| **DetA (Detection Accuracy)** | **86.23%** | 87.15% | 85.45% |
| **DetRe (Detection Recall)** | **86.34%** | 87.30% | 85.53% |
| **DetPr (Detection Precision)** | **99.85%** | 99.79% | **99.90%** |
| **IDF1** | **68.80%** | 77.13% | 70.12% |

### Key Empirical Findings

1. **Association is the Primary Bottleneck:** Detection precision is near-perfect (**99.85%**). Tracking failures are overwhelmingly caused by association breakdown when players cluster.
2. **Scrum Degradation:** In crowded frames ($\text{IoU} \ge 0.20$ or $\ge 18$ players), AssA drops by **11.13 percentage points**. In severe scrums (`SNMOT-116` corner kick), AssA drops to **39.16%**.
3. **Failure Root Causes:** 45.9% of confirmed ID switches are caused by heavy overlap ($\text{IoU} \ge 0.35$), 23.8% by long occlusions ($>20$ frames), and 19.1% by track loss after brief disappearances.
4. **Poor Occlusion Recovery:** ByteTrack recovers original identities in only **39.73%** of occlusion episodes; **60.27%** result in track loss or identity swaps.

---

## 11. Repository Structure

```text
MotionXI/
├── courtvision/                  # Core package
│   ├── detection/                # YOLOv11m detector wrapper
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── tracking/                 # ByteTrack wrapper, evaluation & types
│   │   ├── __init__.py
│   │   ├── bytetrack_tracker.py
│   │   ├── evaluation.py
│   │   └── track_types.py
│   ├── projection/               # Pitch homography projection
│   │   ├── __init__.py
│   │   └── homography.py
│   └── utils/                    # Video decoding & geometry utilities
│       ├── __init__.py
│       └── video.py
├── scripts/                      # Reproducible execution scripts
│   ├── calibrate_pitch.py        # Homography calibration tool
│   ├── evaluate_tracking.py      # Official SoccerNet/TrackEval runner
│   ├── run_detection.py          # Milestone 1 detection CLI
│   └── run_tracking.py           # Milestone 2 ByteTrack CLI
├── tests/                        # Automated pytest suite (32 tests)
│   ├── test_detector.py
│   ├── test_evaluation.py
│   ├── test_homography.py
│   ├── test_packaging.py
│   ├── test_tracker.py
│   └── test_video.py
├── data/                         # Datasets & calibration
│   ├── homography_matrix.json    # Calibrated 18-yard box matrix
│   └── .gitkeep
├── weights/                      # Model weights (best.pt placed here)
│   └── .gitkeep
├── outputs/                      # Generated pipeline outputs
│   └── .gitkeep
├── .env.example                  # Environment configuration template
├── .gitignore                    # Comprehensive git exclusions
├── .python-version               # Python 3.12 pin
├── pyproject.toml                # Project metadata & UV dependencies
├── uv.lock                       # Deterministic UV lockfile
└── README.md                     # Project documentation
```

---

## 12. Current Limitations

1. **ByteTrack Scrum Identity Swapping:** Conventional Kalman filters assume linear velocity and diverge during multi-player scrums.
2. **Single-Quadrant Homography Scope:** The current homography matrix was calibrated from an 18-yard penalty box; coordinates beyond the penalty box are subject to planar perspective extrapolation.
3. **No Learned Re-ID:** Current baseline uses geometric IoU association without appearance feature embeddings.

---

## 13. Upcoming Milestone: Kinematic Memory Tracker (KMT)

Milestone 4 will integrate the **Kinematic Memory Module** to directly address the failure modes identified in Milestone 3A:
- **Momentum Vector Caching:** Caches pre-collision velocity ($\vec{v}$) and acceleration ($\vec{a}$) before $\text{IoU} \ge 0.35$ contact.
- **Ghost Trajectory Projection:** Projects physical trajectories during occlusions ($6 \le \Delta t \le 25$ frames) instead of relying on ambiguous bounding boxes.
- **Physics-Gated Re-Emergence:** Constrains Hungarian association costs with kinematic consistency gates to prevent cross-player swaps.
- **Hypothesis Target:** Reduce IDSW by $\ge 25\%$ and improve AssA by $\ge 5$ percentage points on the SoccerNet benchmark.
