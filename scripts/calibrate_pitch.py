import cv2
import numpy as np
import json
import os

# Standard FIFA 18-Yard Box Dimensions (105m x 68m pitch):
# Depth: 16.5m from goal line (X = 0 to 16.5)
# Width: 40.32m -> Far edge Y = 13.84m, Near edge Y = 54.16m
PITCH_POINTS = np.array([
    [0.0, 13.84],    # 1. Far corner on GOAL LINE
    [16.5, 13.84],   # 2. Far corner out on PITCH
    [16.5, 54.16],   # 3. Near corner out on PITCH
    [0.0, 54.16]     # 4. Near corner on GOAL LINE
], dtype=np.float32)

clicked_points = []
current_frame = None

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        labels = [
            "1: Far corner on GOAL LINE",
            "2: Far corner out on PITCH",
            "3: Near corner out on PITCH",
            "4: Near corner on GOAL LINE"
        ]
        print(f"Logged {labels[len(clicked_points) - 1]} at pixel ({x}, {y})")

def on_trackbar(val, cap):
    global current_frame, clicked_points
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        current_frame = frame
        clicked_points.clear()  # reset clicks if you move the timeline

def main():
    global current_frame, clicked_points
    video_path = "data/test_match.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("Error: Could not read frames from video.")
        return

    ret, current_frame = cap.read()
    window_name = "Scrub Timeline & Click 4 Box Corners"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    # Add interactive timeline slider
    cv2.createTrackbar("Timeline", window_name, 0, total_frames - 1, lambda val: on_trackbar(val, cap))

    print("\n--- TIMELINE CALIBRATOR ---")
    print("1. Drag the 'Timeline' slider at the top to find the exact frame where the box is clearest.")
    print("2. Click the 4 corners of the 18-yard box clockwise (1 -> 2 -> 3 -> 4).")
    print("3. Press 'c' to calculate and save the homography matrix.")
    print("4. Press 'r' to reset clicks on the current frame, or 'q' to quit.\n")

    while True:
        if current_frame is not None:
            display = current_frame.copy()
            for idx, pt in enumerate(clicked_points):
                cv2.circle(display, tuple(pt), 6, (0, 0, 255), -1)
                cv2.putText(display, f"P{idx + 1}", (pt[0] + 10, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
            clicked_points.clear()
            print("Points reset on current frame.")
        elif key == ord('c'):
            if len(clicked_points) == 4:
                break
            print(f"Please click all 4 corners first (currently {len(clicked_points)}).")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    src_pts = np.array(clicked_points, dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, PITCH_POINTS)

    os.makedirs("data", exist_ok=True)
    with open("data/homography_matrix.json", "w") as f:
        json.dump(H.tolist(), f)

    print("\nSUCCESS! Homography matrix saved to data/homography_matrix.json")
    print(H)

if __name__ == "__main__":
    main()