# CourtVision AI (AthletaTrack) — Team Handover

Practical onboarding and operational handover for CourtVision AI. Use this guide to set up the environment, understand completed milestones and verified baseline results, and take ownership of upcoming development tasks.

---

## 1. Project Overview

### Purpose & End-to-End Pipeline
CourtVision AI is an automated football analytics pipeline designed for broadcast match footage. The system takes raw 1080p video as input and produces player tracks, pitch-projected coordinates, and tactical analytics:
1. **Player Detection:** Identifies player bounding boxes and estimates foot-contact points using YOLOv11m.
2. **Multi-Object Tracking:** Assigns consistent player IDs across video frames (ByteTrack baseline; Kinematic Memory Tracker target).
3. **Pitch Projection:** Maps image-space foot positions $(u, v)$ to real-world metric pitch coordinates $(X, Y)$ on a standard FIFA pitch ($105\text{m} \times 68\text{m}$).
4. **Physical & Tactical Analytics:** Computes total distance covered, top speed, sprint counts ($>25\text{ km/h}$), 2D kernel density estimation (KDE) heatmaps, and tactical match ratings.

```
Raw Match Video (1080p, 25 FPS)
        │
        ▼
[ YOLOv11m Detector ] ──► Bounding Boxes & Foot Contacts
        │
        ▼
[ Multi-Object Tracking ] (ByteTrack Baseline / KMT Target)
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
[ Pitch Homography ]                    [ Visual Analytics ]
  Image (u,v) ──► Pitch (X,Y) in meters    Kit Color (HSV) ──► Teams
        │                                        │
        └───────────────────┬────────────────────┘
                            ▼
              [ Physical & Tactical Engine ]
         Distance, Speed, Sprints, 2D Heatmaps
                            │
                            ▼
              [ REST API & Web Dashboard ]
```

### The Core Research Problem
Broadcast football tracking suffers from frequent identity switches (IDSW) and track fragmentation during dense player interactions:
- **Cluster Collisions:** During corner kicks, set pieces, and sliding tackles, player bounding boxes heavily overlap ($\text{IoU} \ge 0.35$).
- **Visual Ambiguity:** Similar team kits and close physical contact degrade appearance embeddings.
- **Kalman Filter Breakdown:** Standard linear constant-velocity assumptions fail when players collide, decelerate sharply, or occlude one another. When an occluded player emerges, conventional trackers often assign a new track ID or swap identities between teammates.

### The Proposed Solution: Kinematic Memory Tracker (KMT)
Kinematic Memory Tracker (KMT) is our physics-grounded research extension designed to eliminate cluster ID switches:
- **Momentum Caching:** Detects when players enter a high-density cluster ($\text{IoU} \ge 0.35$) and stores their entry velocity ($\vec{v}$) and acceleration ($\vec{a}$) before visual features become corrupted.
- **Ghost Trajectories:** Extrapolates ballistic trajectories through occlusions lasting 6 to 25 frames, maintaining an estimated kinematic state while the player is hidden.
- **Kinematic Consistency Gating:** When an occluded player reappears, KMT restricts Hungarian association to candidates whose position and velocity match physical momentum constraints, rejecting non-physical trajectory jumps.

### Baseline vs. Research Target
- **ByteTrack Baseline:** Serves as the frozen, unmodified control tracker. It has been evaluated against ground truth to document exact failure modes.
- **KMT Target:** To be evaluated against ByteTrack on identical video sequences, detection inputs, and metrics to isolate and measure the tracking performance gain.

### Component Implementation Status
| Component | Status | Implementation Details |
|---|---|---|
| **YOLOv11m Detector** | **Completed** | Model checkpoint `weights/best.pt` (~39 MB), 93.1% mAP@50, MPS/CUDA/CPU dispatch. |
| **ByteTrack Baseline** | **Completed** | SuperVision ByteTrack wrapper, bottom-center foot contact points, 15-frame history trails. |
| **SoccerNet GT Evaluation** | **Completed** | Automated benchmark using `trackeval` over 5 held-out sequences (3,750 frames). |
| **Failure Mode Audit** | **Completed** | Quantified ID switches, cluster collapses, and occlusion recovery rates. |
| **4-Point Pitch Homography** | **Baseline Available** | `scripts/calibrate_pitch.py`, `data/homography_matrix.json`, boundary validation logic. |
| **Kinematic Memory Tracker** | **Implementation Available; Evaluation Pending** | `courtvision/tracking/kmt_tracker.py` implements image-space momentum caching, cluster states, ghost predictions, and speed-gated Hungarian association. Select it with `scripts/run_tracking.py --tracker kmt`; unit tests are added but have not yet been run. |
| **Kit Color Clustering** | **Pending** | Upper-body HSV K-means clustering for team assignment. |
| **Physical Analytics Engine** | **Pending** | Sprint detection ($>25\text{ km/h}$), distance covered, 2D KDE heatmaps. |
| **FastAPI & Dashboard** | **Pending** | REST backend and interactive tactical dashboard. |

---

## 2. Current Status & Verified Results

### Detector Performance
The custom YOLOv11m detector was trained on approximately 14,000 frames from the SoccerNet Tracking dataset. Evaluation on held-out validation images confirms strong detection capability:
- **mAP@50:** **93.1%**
- **Precision:** **95.4%**
- **Recall:** **91.6%**

Local inference runs efficiently on Apple Silicon (M4 MPS) as well as NVIDIA CUDA and CPU backends via automatic device selection.

### Test Match Sequence (`data/test_match.mp4`)
An initial 600-frame broadcast test clip was evaluated with YOLOv11m and ByteTrack:
- **Frames Decoded:** 600 frames (24.0 seconds at 25 FPS, 1080p).
- **Detections Generated:** 7,626 total player detections (mean 12.71 players/frame).
- **Unique ByteTrack IDs:** 115 track IDs created.
- **Crowded Frame Candidates:** 406 frames with overlapping bounding boxes.
- **Heuristic Switch Candidates:** 199 suspected identity-switch events.

### Ground-Truth Benchmark Results (SoccerNet Tracking 2023)
ByteTrack was formally evaluated on **5 held-out test sequences** (`SNMOT-116` to `SNMOT-120`, 3,750 frames total @ 25 FPS) using the official `trackeval` evaluation library:

| Evaluation Metric | Overall (3,750 frames) | Normal Subset | Crowded Subset ($\text{IoU} \ge 0.20$ or $\ge 18$ players) |
|---|---:|---:|---:|
| **HOTA** | **72.81%** | 79.67% | 72.62% |
| **AssA (Association Accuracy)** | **61.48%** | **72.84%** | **61.71%** |
| **MOTA** | **83.26%** | 84.01% | 82.33% |
| **ID Switches (TrackEval CLEAR MOT)** | **294** | 144 | **231** |
| **Fragmentations** | **649** | 245 | **452** |
| **DetA (Detection Accuracy)** | **86.23%** | 87.15% | 85.45% |
| **DetRe (Detection Recall)** | **86.34%** | 87.30% | 85.53% |
| **DetPr (Detection Precision)** | **99.85%** | 99.79% | **99.90%** |
| **IDF1** | **68.80%** | 77.13% | 70.12% |

### Key Metric Definitions
- **mAP@50:** Mean Average Precision at an Intersection-over-Union threshold of 0.50.
- **HOTA (Higher Order Tracking Accuracy):** Overall tracking metric that explicitly balances detection accuracy and association accuracy.
- **AssA (Association Accuracy):** Measures how accurately the tracker maintains correct identity linkages across time.
- **MOTA (Multi-Object Tracking Accuracy):** Traditional MOT metric combining false positives, missed detections, and identity switches.
- **ID Switch (IDSW):** Occurs when a tracker switches the identities of two tracked players.
- **Fragmentation (Frag):** Occurs when an active track is interrupted into separate disjoint segments.

### Key Empirical Findings
1. **Detection is Highly Reliable:** Detection precision is 99.85% and recall is 86.34%. Tracking failures are overwhelmingly caused by association breakdown rather than missed detections.
2. **Crowding Degrades Association:** In crowded frames, AssA drops by 11.13 percentage points (from 72.84% down to 61.71%). On corner kick sequences (`SNMOT-116`), AssA collapses to **39.16%**.
3. **Occlusion Recovery Fails:** Across 365 observed occlusion events lasting 3 or more frames, ByteTrack recovered the correct identity in only **39.73%** of cases. In the remaining 60.27%, it assigned a new track ID or swapped identities.

### ID Switch Discrepancy (294 vs. 320)
You will observe two different ID switch counts in project logs:
- **TrackEval Benchmark (294 IDSW):** Measures active identity transfers between simultaneous ground-truth targets using CLEAR MOT standards (`outputs/evaluation/metrics.json`).
- **Custom Failure Audit (320 switches):** Measures total identity loss events across track lifetimes (`outputs/evaluation/id_switch_events.csv`). This includes 76 long-occlusion track terminations ($>20$ frames) and 61 brief disappearances ($>3$ frames) where ByteTrack failed to re-associate the player and spawned a duplicate ID.
- **Rule for Reporting:** Both numbers are legitimate and must be reported clearly. Use 294 for formal benchmark comparisons; use 320 to illustrate track fragmentation and re-association failures.

---

## 3. Homography and Pitch Calibration

### What It Does
Pitch homography computes a $3 \times 3$ perspective transformation matrix ($H$) that maps a player's 2D image position (foot-contact pixel $u, v$) into real-world pitch coordinates ($X, Y$ in meters) on a standardized FIFA pitch ($105.0\text{m} \times 68.0\text{m}$).

$$\begin{bmatrix} X' \\ Y' \\ Z' \end{bmatrix} = H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad X = \frac{X'}{Z'}, \quad Y = \frac{Y'}{Z'}$$

### What Currently Exists in the Codebase
- **Interactive Calibration Tool (`scripts/calibrate_pitch.py`):** An OpenCV script with a video scrubber trackbar. The operator scrubs to a frame where the 18-yard box is clear, clicks the 4 corners of the penalty box, and saves the resulting matrix to `data/homography_matrix.json`.
- **Baseline Matrix (`data/homography_matrix.json`):** Tracked $3 \times 3$ homography matrix computed from the 18-yard box of `data/test_match.mp4`.
- **Projection Module (`courtvision/projection/homography.py`):** Contains `PitchHomography`, providing:
  - `project_point(u, v)`: Converts single pixel coordinates to pitch meters $(X, Y)$ and checks whether the point falls within the $105\text{m} \times 68\text{m}$ pitch.
  - `project_bbox_foot(x1, y1, x2, y2)`: Extracts the bottom-center coordinate $\left(\frac{x1+x2}{2}, y2\right)$ and projects it to pitch coordinates.
  - `is_within_pitch(X, Y)` and `is_within_calibrated_zone(X, Y)`: Flags coordinates outside the pitch or far from the calibrated box.
  - `project_points(points_array)`: Vectorized projection for batch arrays.
- **Automated Tests (`tests/test_homography.py`):** 6 tests validating matrix loading, point projection, boundary checks, and foot-contact logic.

### Current Limitations of the Baseline Calibration
1. **Local Calibration Zone:** The current matrix was calibrated using only 4 corners of the left 18-yard penalty box ($16.5\text{m} \times 40.32\text{m}$). Within this penalty box, reprojection error is minimal ($\approx 0.05\text{m}$).
2. **Planar Extrapolation Divergence:** Planar homography becomes mathematically unstable when extrapolating far beyond the calibrated region. Projecting midfield or far-touchline pixels using this matrix produces catastrophic errors (e.g., coordinates projecting hundreds of meters outside the stadium).
3. **Never Assume Full-Pitch Accuracy:** The current calibration cannot be assumed accurate across the whole pitch without empirical validation.

### Why Calibration Accuracy Matters
Physical analytics depend entirely on metric pitch coordinates:
- If $(X, Y)$ coordinates are distorted, distances traveled will be wildly exaggerated or compressed.
- Velocity calculations ($\frac{\Delta d}{\Delta t}$) will produce artificial speed spikes, corrupting sprint counts ($>25\text{ km/h}$).
- 2D heatmaps will warp and bleed outside the pitch boundaries.

### Camera Movement: Pan, Tilt, and Zoom (PTZ)
In broadcast football, the main camera continuously pans, tilts, and zooms to follow the ball. A single static homography matrix is valid **only** while the camera remains stationary. 

Our camera motion analysis reveals:
- **Static Set Pieces (e.g., `SNMOT-116`):** The camera remains locked on the corner flag. Cumulative camera shift is $<16$ pixels over 150 frames. A single static homography remains valid throughout the clip.
- **Dynamic Action (e.g., `test_match.mp4` and `SNMOT-117`):** The camera pans rapidly across the pitch, accumulating over 1,200 to 2,600 pixels of lateral displacement. Under these conditions, a static homography shifts projected pitch coordinates by tens of meters within seconds.
- **Requirement:** Member 3 must investigate whether the system requires per-shot homography switching, optical flow camera compensation, or dynamic landmark-based recalibration.

### Validation Rules and Scientific Discipline
1. **Preserve Baseline:** Do not overwrite or delete `data/homography_matrix.json`. Keep it as the verified 4-point baseline.
2. **Multi-Region Landmark Validation:** Validate projections against known pitch landmarks (halfway line, center circle, penalty spots, corner flags) where visible.
3. **Reject Out-of-Bounds Points:** Out-of-pitch projections must be flagged and filtered out.
4. **No False Claims:** **Do NOT claim accurate real-world distances, speeds, or heatmaps until the projection pipeline has been validated with measured error bounds.**

---

## 4. Repository Map

| Path | Purpose & Responsibilities |
|---|---|
| `courtvision/detection/` | `detector.py`: YOLOv11m wrapper, foot contact calculation, device dispatch (MPS/CUDA/CPU). |
| `courtvision/tracking/` | `bytetrack_tracker.py`: SuperVision ByteTrack baseline. `evaluation.py`: TrackEval benchmark harness. `track_types.py`: Track data structures. |
| `courtvision/projection/` | `homography.py`: `PitchHomography` class mapping pixels $(u, v)$ to pitch meters $(X, Y)$ with boundary validation. |
| `courtvision/utils/` | `video.py`: Video decoding, frame-count validation, and pairwise bounding box IoU calculation. |
| `scripts/run_detection.py` | Runs YOLOv11m detection on video; exports CSV/JSON detections and annotated video. |
| `scripts/run_tracking.py` | Runs ByteTrack baseline; exports tracking CSV, crowding metrics, and visualization video. |
| `scripts/evaluate_tracking.py` | Runs TrackEval against SoccerNet GT annotations; generates metrics and failure logs. |
| `scripts/calibrate_pitch.py` | Interactive OpenCV UI for scrubbing video and clicking 4 penalty-box points to save homography. |
| `tests/` | 32 automated tests covering detector, tracker, video, homography, evaluation, and UV packaging. |
| `data/` | `homography_matrix.json` (tracked). Local video and benchmark folders are gitignored. |
| `weights/` | `weights/.gitkeep` (tracked). Model checkpoint `best.pt` (~39 MB) is gitignored. |
| `outputs/` | `outputs/.gitkeep` (tracked). Local directory for generated videos, logs, and evaluation metrics. |
| `pyproject.toml` / `uv.lock` | UV project configuration, build settings, and locked dependency tree. |

---

## 5. Setup and Running the Project

Use **UV** exclusively. Do NOT use `pip`, `venv`, `virtualenv`, `conda`, or `poetry`.

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Ad1lSy3d/MotionXI.git
cd MotionXI

# Sync dependencies exactly according to uv.lock
uv sync

# Run test suite to verify installation (all 32 tests must pass)
uv run pytest -v
```

### 2. Local Asset Requirements (Excluded from Git)
These files are excluded from Git via `.gitignore` to prevent large binary blobs in source control:
- **YOLOv11m Weights (`weights/best.pt`, ~39 MB):** Obtain from team storage and place at `weights/best.pt`.
- **Demo Video (`data/test_match.mp4`, ~8.0 MB):** Obtain from team storage and place at `data/test_match.mp4`.
- **SoccerNet Benchmark (`data/soccernet_tracking/`):** Place sequences `SNMOT-116` through `SNMOT-120` inside:
  ```
  data/soccernet_tracking/
  ├── annotations/
  │   ├── SNMOT-116/gt/gt.txt
  │   ├── SNMOT-117/gt/gt.txt
  │   └── ...
  └── videos/
      ├── SNMOT-116.mp4
      ├── SNMOT-117.mp4
      └── ...
  ```

### 3. Verified Execution Commands
All commands run via `uv run` to ensure isolation within the project virtual environment:

```bash
# 1. Run Player Detection
uv run python scripts/run_detection.py \
  --video data/test_match.mp4 \
  --weights weights/best.pt \
  --output-dir outputs/detection

# 2. Run ByteTrack Baseline Tracking
uv run python scripts/run_tracking.py \
  --video data/test_match.mp4 \
  --weights weights/best.pt \
  --output-dir outputs/tracking

# 3. Run Benchmark Tracking Evaluation against SoccerNet Ground Truth
uv run python scripts/evaluate_tracking.py \
  --ground-truth data/soccernet_tracking/annotations \
  --videos-dir data/soccernet_tracking/videos \
  --weights weights/best.pt \
  --output-dir outputs/evaluation
```

---

## 6. Division of Responsibilities & Task Ownership

To prevent merge conflicts and preserve research discipline, the three team members work in parallel across decoupled modules.

### Decoupling Rule: Homography Does NOT Block KMT
**Homography is NOT a prerequisite for KMT.**
- KMT operates purely in **image space** (pixel coordinates, bounding box velocities $\Delta u/\Delta t, \Delta v/\Delta t$, and visual history). Its objective is to prevent identity switches during visual occlusions.
- Homography operates downstream, mapping tracks into real-world pitch coordinates for physical analytics.
- Member 1 and Member 2 can implement, evaluate, and benchmark KMT without waiting for pitch calibration improvements. Member 3 develops pitch calibration and camera motion analysis in parallel.

```
[ Member 1: KMT Development ]  ──► [ Member 2: Benchmark Evaluation ]
  (Image-Space Tracking)               (HOTA, AssA, IDSW Comparison)
         ▲
         │ Decoupled: KMT does not depend on Homography
         ▼
[ Member 3: Calibration & Motion ] ──► [ Downstream Analytics Engine ]
  (Pitch Projection, PTZ Motion)           (Distance, Speed, Heatmaps)
```

### Detailed Member Responsibilities

| Role | Team Member | Primary Responsibilities | Deliverables & Handoffs |
|---|---|---|---|
| **Tracking Research** | **Member 1** | 1. Implemented `kmt_tracker.py` under `courtvision/tracking/` with momentum caching, ghost predictions, and speed-gated association.<br>2. Added `--tracker kmt` to the tracking CLI and dashboard.<br>3. Keep `bytetrack_tracker.py` unchanged as control.<br>4. Run the new focused tests and real-video smoke test once assets are available. | **Handoff to Member 2:** KMT implementation is ready for benchmark integration; its focused tests have not yet been run. |
| **Evaluation & Benchmark** | **Member 2** | 1. Maintain and lock evaluation protocol (`scripts/evaluate_tracking.py`).<br>2. Add tracker selection to the evaluator, then benchmark KMT against ByteTrack on the 5 SoccerNet sequences.<br>3. Run ablation experiments on memory window ($\Delta t \in [5, 15, 30]$ frames) and kinematic gating thresholds.<br>4. Generate final comparison tables (HOTA, AssA, MOTA, IDSW, Frag, FPS). | **Handoff to Team:** Verified experimental evidence showing whether KMT outperforms ByteTrack. |
| **Calibration & Analytics** | **Member 3** | 1. Audit and validate pitch homography across multiple pitch regions.<br>2. Investigate full-pitch calibration using additional pitch markings (halfway line, center circle, penalty spots).<br>3. Analyze camera PTZ dynamics and evaluate per-shot / dynamic recalibration.<br>4. Build kit color clustering (upper-body HSV K-means) for team separation.<br>5. Prototype physical analytics (speeds, distances, heatmaps) in `courtvision/analytics/`. | **Handoff to Team:** Validated pitch coordinate transformation with documented error bounds. |

---

## 7. Immediate Next Steps & Collaboration Rules

### Collaboration Guidelines
1. **Branch Hygiene:** Work in dedicated feature branches:
   - Member 1: `feat/kmt-tracker`
   - Member 2: `feat/kmt-eval-ablation`
   - Member 3: `feat/pitch-calibration-analytics`
2. **Interface Agreement:** Member 1 and Member 2 agree on standard tracker inputs and outputs (0-indexed video frames, 1-indexed MOT ground truth format).
3. **Evaluation Protocol Lockdown:** Member 2 locks the evaluation code before evaluating KMT to prevent accidental hyperparameter tuning to test sequences.
4. **Git Discipline:** Never commit video files, weights, benchmark datasets, or temporary outputs to Git.

### Immediate Action Checklist
- [x] **Git Remote Verified:** `origin/main` is up-to-date with remote repository.
- [ ] **Teammate Onboarding:** Members 2 & 3 clone `MotionXI`, run `uv sync`, and confirm the automated test suite with `uv run pytest -v`.
- [ ] **Asset Distribution:** Place `weights/best.pt`, `data/test_match.mp4`, and `data/soccernet_tracking/` in local workspaces.
- [ ] **Branch Creation:** Create `feat/kmt-tracker`, `feat/kmt-eval-ablation`, and `feat/pitch-calibration-analytics`.
- [x] **Member 1 (KMT):** Implement tracker module, CLI selection, and unit-test cases; focused tests and real-video validation remain to be run.
- [ ] **Member 2 (Eval):** Add KMT selection to `evaluate_tracking.py`, verify baseline outputs, and benchmark against ByteTrack.
- [ ] **Member 3 (Calibration):** Begin landmark reprojection error tests on visible pitch markings and document camera PTZ drift.
