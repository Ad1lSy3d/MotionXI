"""
scripts/calibrate_pitch_full.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interactive full-pitch homography calibration for CourtVision AI.

Usage
-----
    uv run python scripts/calibrate_pitch_full.py [--video data/test_match.mp4] \\
        [--output data/homography_full_pitch.json]

Workflow
--------
1. The video opens with a frame scrubber slider. Navigate to a frame where
   pitch landmarks are clearly visible and unoccluded.
2. Select a landmark from the on-screen menu by pressing its unique single-character hotkey.
3. Click on the exact pixel location of that landmark in the frame.
4. Repeat for landmarks distributed across both halves and touchlines (>= 10 recommended).
5. Press 'e' to estimate the homography and preview the validation report.
6. Press 's' to save the calibration to the output JSON path.
7. Press 'r' to undo the last clicked landmark. Press 'q' to quit without saving.

Coordinate Convention
---------------------
Origin (0, 0): left goal-line / near touchline.
X axis: pitch length, 0 → 105 m (left goal → right goal).
Y axis: pitch width,  0 → 68 m  (near touchline → far touchline).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from courtvision.projection.calibrator import (
    PITCH_LANDMARKS,
    HomographyCalibrator,
    MIN_CORRESPONDENCES,
    RECOMMENDED_CORRESPONDENCES,
)


# ── Selectable landmarks with unique, single-character, non-conflicting hotkeys ──
# Reserved control keys: 'e' (estimate), 's' (save), 'r' (undo), 'q' (quit)

MENU_ITEMS = [
    ("1", "left_goal_line_near_touch",  "Left goal-line + near touchline (0, 0)"),
    ("2", "left_goal_line_far_touch",   "Left goal-line + far touchline  (0, 68)"),
    ("3", "corner_near_right",          "Right goal-line + near touchline (105, 0)"),
    ("4", "corner_far_right",           "Right goal-line + far touchline  (105, 68)"),
    ("5", "left_goal_line_near_pa",     "Left PA near on goal-line        (0, 13.84)"),
    ("6", "left_goal_line_far_pa",      "Left PA far on goal-line         (0, 54.16)"),
    ("7", "left_pa_near_corner",        "Left PA near pitch corner        (16.5, 13.84)"),
    ("8", "left_pa_far_corner",         "Left PA far pitch corner         (16.5, 54.16)"),
    ("9", "halfway_near_touch",         "Halfway + near touchline         (52.5, 0)"),
    ("0", "halfway_far_touch",          "Halfway + far touchline          (52.5, 68)"),
    ("a", "center_spot",                "Center spot                      (52.5, 34)"),
    ("b", "right_pa_near_corner",       "Right PA near pitch corner       (88.5, 13.84)"),
    ("c", "right_pa_far_corner",        "Right PA far pitch corner        (88.5, 54.16)"),
    ("d", "right_goal_line_near_pa",    "Right PA near on goal-line       (105, 13.84)"),
    ("f", "right_goal_line_far_pa",     "Right PA far on goal-line        (105, 54.16)"),
    ("g", "left_penalty_spot",          "Left penalty spot                (11, 34)"),
    ("h", "right_penalty_spot",         "Right penalty spot               (94, 34)"),
]

HOTKEY_TO_LABEL = {hk: lbl for hk, lbl, _ in MENU_ITEMS}
HOTKEY_TO_DESC  = {hk: dsc for hk, _, dsc in MENU_ITEMS}


# ── State ───────────────────────────────────────────────────────────────────

class CalibrationState:
    def __init__(self, calibrator: HomographyCalibrator):
        self.cal = calibrator
        self.current_landmark_key: Optional[str] = None
        self.last_result = None

    def select_landmark(self, hotkey: str) -> None:
        if hotkey not in HOTKEY_TO_LABEL:
            return
        self.current_landmark_key = hotkey
        lbl = HOTKEY_TO_LABEL[hotkey]
        pitch_xy = PITCH_LANDMARKS[lbl]
        print(f"\n[SELECT] '{hotkey}': {HOTKEY_TO_DESC[hotkey]} → pitch {pitch_xy}")
        print("  Click on the corresponding pixel in the frame.")

    def commit_click(self, px: int, py: int) -> None:
        if self.current_landmark_key is None:
            print("[WARN] No landmark selected. Press a landmark hotkey first.")
            return
        lbl = HOTKEY_TO_LABEL[self.current_landmark_key]
        pitch_x, pitch_y = PITCH_LANDMARKS[lbl]
        self.cal.add_correspondence(px, py, pitch_x, pitch_y, label=lbl)
        n = len(self.cal.correspondences)
        print(f"  [ADDED] {lbl}: pixel=({px}, {py}) → pitch=({pitch_x}, {pitch_y}) [total: {n}]")
        self.current_landmark_key = None

    def undo_last(self) -> None:
        corr = self.cal.correspondences
        if not corr:
            print("[INFO] No correspondences to undo.")
            return
        removed = corr[-1]
        self.cal.clear_correspondences()
        for c in corr[:-1]:
            self.cal.add_correspondence(
                c.pixel_x, c.pixel_y, c.pitch_x, c.pitch_y, c.label
            )
        print(f"[UNDO] Removed {removed.label} pixel=({removed.pixel_x}, {removed.pixel_y})")

    def estimate(self) -> None:
        n = len(self.cal.correspondences)
        if n < MIN_CORRESPONDENCES:
            print(f"[ERROR] Need >= {MIN_CORRESPONDENCES} correspondences (have {n}).")
            return
        result = self.cal.estimate()
        self.last_result = result
        print("\n" + "=" * 65)
        print("CALIBRATION ESTIMATE REPORT")
        print(f"  Scope               : {result.calibration_scope}")
        print(f"  Inliers             : {result.inlier_count} / {result.total_correspondences}")
        print(f"  Inlier RMS error    : {result.rms_reprojection_px:.2f} px  |  {result.rms_reprojection_m:.3f} m")
        print(f"  Inlier Max error    : {result.max_reprojection_px:.2f} px  |  {result.max_reprojection_m:.3f} m")
        print(f"  Overall RMS error   : {result.overall_rms_reprojection_px:.2f} px  |  {result.overall_rms_reprojection_m:.3f} m")
        print(f"  Coverage score      : {result.coverage_score:.3f}")
        print(f"  Calibrated zone (m) : X {result.calibrated_zone_m['x_range']}, Y {result.calibrated_zone_m['y_range']}")
        print(f"  Quadrant coverage   : {result.quadrant_coverage}")
        print(f"  Valid homography    : {result.is_valid}")
        if result.warnings:
            print("\n  Validation Warnings:")
            for w in result.warnings:
                print(f"    !  {w}")
        print("=" * 65 + "\n")


def mouse_callback(event, x, y, flags, state: CalibrationState):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.commit_click(x, y)


def draw_overlay(frame, state: CalibrationState) -> None:
    corr = state.cal.correspondences
    for i, c in enumerate(corr):
        pt = (int(c.pixel_x), int(c.pixel_y))
        cv2.circle(frame, pt, 6, (0, 255, 80), -1)
        cv2.circle(frame, pt, 7, (255, 255, 255), 1)
        label = c.label or f"P{i+1}"
        cv2.putText(frame, label[:16], (pt[0] + 9, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1)

    status_lines = [
        f"Correspondences: {len(corr)} / {RECOMMENDED_CORRESPONDENCES} recommended",
        f"Selected landmark: {HOTKEY_TO_DESC.get(state.current_landmark_key, 'None (press hotkey)')}",
        "Controls: [e]=estimate  [s]=save  [r]=undo  [q]=quit",
    ]
    for row, line in enumerate(status_lines):
        cv2.putText(frame, line, (10, 22 + row * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-pitch homography calibration")
    parser.add_argument("--video",  default="data/test_match.mp4", help="Video file")
    parser.add_argument("--output", default="data/homography_full_pitch.json",
                        help="Output calibration JSON path")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    print("\n=== CourtVision Full-Pitch Landmark Calibrator ===")
    print("Available landmark hotkeys:")
    for hk, lbl, desc in MENU_ITEMS:
        pitch_xy = PITCH_LANDMARKS.get(lbl, "?")
        print(f"  [{hk}] {desc}  ->  {pitch_xy}")
    print()

    cap = cv2.VideoCapture(args.video)
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame from video.")
        sys.exit(1)

    calibrator = HomographyCalibrator()
    state = CalibrationState(calibrator)

    win = "CourtVision Full-Pitch Calibrator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse_callback, state)
    cv2.createTrackbar("Frame", win, 0, total_frames - 1,
                       lambda v: _seek(cap, v, state))

    current_frame = [frame]

    def _seek(cap, val, state):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        ret, f = cap.read()
        if ret:
            current_frame[0] = f
            state.current_landmark_key = None

    while True:
        display = current_frame[0].copy()
        draw_overlay(display, state)
        cv2.imshow(win, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            state.undo_last()
        elif key == ord('e'):
            state.estimate()
        elif key == ord('s'):
            if state.last_result is None:
                print("[WARN] Run estimate first (press 'e').")
            elif not state.last_result.is_valid:
                print("[WARN] Calibration is not valid — review warnings before saving.")
            else:
                path = calibrator.save(args.output)
                print(f"\n[SAVED] Calibration written to {path}")
                print(f"  Scope          : {state.last_result.calibration_scope}")
                print(f"  Inlier RMS px  : {state.last_result.rms_reprojection_px:.2f}")
                print(f"  Calibrated zone: {state.last_result.calibrated_zone_m}")
        else:
            char_key = chr(key) if 0 <= key < 128 else ""
            if char_key in HOTKEY_TO_LABEL:
                state.select_landmark(char_key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
