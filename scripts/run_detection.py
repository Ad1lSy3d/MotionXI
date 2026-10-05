import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from courtvision.detection.detector import Detection, PlayerDetector
from courtvision.projection.homography import PitchHomography
from courtvision.utils.video import (
    VideoMetadata,
    create_video_writer,
    draw_detections,
    get_video_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CourtVision AI - Milestone 1: YOLOv11m Full-Video Inference Pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/test_match.mp4",
        help="Path to input video file (default: data/test_match.mp4)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to YOLOv11m trained weights (default: weights/best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/detection",
        help="Directory to save output files (default: outputs/detection)",
    )
    parser.add_argument(
        "--homography",
        type=str,
        default="data/homography_matrix.json",
        help="Path to calibrated 3x3 homography matrix (default: data/homography_matrix.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for PyTorch inference (mps, cuda, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip generating annotated video output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)
    homography_path = Path(args.homography)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("COURTVISION AI: MILESTONE 1 DETECTION PIPELINE")
    print("==================================================")
    print(f"Input Video:      {video_path}")
    print(f"Model Checkpoint: {weights_path}")
    print(f"Output Directory: {output_dir}")

    # 1. Read and validate video metadata
    metadata: VideoMetadata = get_video_metadata(video_path)
    print(f"Video Properties: {metadata.width}x{metadata.height} @ {metadata.fps:.2f} FPS")
    print(f"Total Frames:     {metadata.frame_count} ({metadata.duration_sec:.2f}s)")

    # 2. Validate and initialize Homography Projection utility
    homography: PitchHomography = None
    if homography_path.exists():
        homography = PitchHomography(homography_path)
        print(f"Homography:       Loaded 3x3 matrix from {homography_path}")
        print("                  Standard pitch bounds: [0, 105]m x [0, 68]m")
        print("                  Note: Calibrated from 18-yard box; boundary checks enforced.")
    else:
        print(f"Homography:       [WARNING] {homography_path} not found. Skipping pitch projection.")

    # 3. Initialize Detector
    detector = PlayerDetector(
        model_path=str(weights_path),
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )
    print(f"Detector Device:  {detector.device.upper()}")
    print(f"Conf Threshold:   {detector.conf_threshold}")
    print(f"Image Size:       {detector.imgsz}")

    # 4. Initialize Video Writer if enabled
    annotated_video_path = output_dir / "annotated.mp4"
    video_writer = None
    if not args.no_video:
        video_writer = create_video_writer(
            output_path=annotated_video_path,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
        )
        print(f"Video Output:     {annotated_video_path}")

    # 5. Open video stream
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    all_detections: List[Dict[str, Any]] = []
    inference_times_ms: List[float] = []
    frame_processing_times_sec: List[float] = []

    print("\nProcessing frames...")
    pipeline_start_time = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            dets, infer_ms = detector.detect(
                frame=frame,
                frame_idx=frame_idx,
                fps=metadata.fps,
            )
            inference_times_ms.append(infer_ms)

            # Store detection records
            for d in dets:
                record = d.to_dict()
                if homography is not None:
                    px, py, is_valid = homography.project_point(d.foot_x, d.foot_y)
                    in_box = homography.is_within_calibrated_zone(px, py)
                    record["pitch_x"] = round(px, 3) if not np.isnan(px) else None
                    record["pitch_y"] = round(py, 3) if not np.isnan(py) else None
                    record["pitch_valid"] = bool(is_valid)
                    record["pitch_in_calibrated_zone"] = bool(in_box)
                else:
                    record["pitch_x"] = None
                    record["pitch_y"] = None
                    record["pitch_valid"] = False
                    record["pitch_in_calibrated_zone"] = False
                all_detections.append(record)

            # Draw visual telemetry
            if video_writer is not None:
                annotated_frame = draw_detections(
                    frame=frame,
                    detections=dets,
                    frame_idx=frame_idx,
                    total_frames=metadata.frame_count,
                    fps=metadata.fps,
                    inference_ms=infer_ms,
                    device_name=detector.device,
                )
                video_writer.write(annotated_frame)

            frame_elapsed = time.perf_counter() - frame_start
            frame_processing_times_sec.append(frame_elapsed)

            frame_idx += 1
            if frame_idx % 60 == 0 or frame_idx == metadata.frame_count:
                fps_instant = 1.0 / frame_elapsed if frame_elapsed > 0 else 0.0
                print(
                    f"  Frame {frame_idx:03d}/{metadata.frame_count:03d} | "
                    f"Detections: {len(dets):2d} | "
                    f"Infer: {infer_ms:4.1f} ms | "
                    f"Instant Rate: {fps_instant:4.1f} FPS"
                )

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    pipeline_total_time = time.perf_counter() - pipeline_start_time
    effective_fps = frame_idx / pipeline_total_time if pipeline_total_time > 0 else 0.0
    avg_frame_time_ms = (pipeline_total_time / frame_idx * 1000.0) if frame_idx > 0 else 0.0
    avg_inference_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0

    print("\nInference processing completed successfully.")

    # 6. Save detections CSV
    csv_path = output_dir / "detections.csv"
    if all_detections:
        fieldnames = list(all_detections[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detections)
    else:
        with open(csv_path, "w") as f:
            f.write("")
    print(f"Saved CSV detections: {csv_path}")

    # 7. Save detections JSON
    json_path = output_dir / "detections.json"
    with open(json_path, "w") as f:
        json.dump(all_detections, f, indent=2)
    print(f"Saved JSON detections: {json_path}")

    # 8. Compute detection statistics across the video
    confidences = [d["confidence"] for d in all_detections]
    if confidences:
        mean_conf = float(np.mean(confidences))
        median_conf = float(np.median(confidences))
        min_conf = float(np.min(confidences))
        max_conf = float(np.max(confidences))
        dets_per_frame = float(len(all_detections) / frame_idx) if frame_idx > 0 else 0.0
    else:
        mean_conf = median_conf = min_conf = max_conf = dets_per_frame = 0.0

    # 9. Save run metadata JSON
    metadata = metadata.with_decoded_count(frame_idx)
    metadata_path = output_dir / "run_metadata.json"
    run_metadata: Dict[str, Any] = {
        "pipeline_version": "0.1.0",
        "milestone": "Milestone 1: YOLO Detection Pipeline",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "weights": str(weights_path),
            "architecture": "YOLOv11m",
            "task": "detect",
            "classes": detector.model.names,
            "conf_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold,
            "imgsz": detector.imgsz,
        },
        "hardware": {
            "device": detector.device,
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "input_video": {
            "path": str(video_path),
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "duration_sec": metadata.duration_sec,
            "container_frame_count": metadata.container_frame_count,
            "decoded_frame_count": metadata.decoded_frame_count,
            "frame_count_source": metadata.frame_count_source,
            "frame_count_warning": metadata.frame_count_warning,
        },
        "performance": {
            "frames_processed": frame_idx,
            "total_processing_time_sec": round(pipeline_total_time, 3),
            "effective_pipeline_fps": round(effective_fps, 2),
            "avg_frame_processing_ms": round(avg_frame_time_ms, 2),
            "avg_model_inference_ms": round(avg_inference_ms, 2),
        },
        "detection_statistics": {
            "total_detections": len(all_detections),
            "detections_per_frame": round(dets_per_frame, 2),
            "confidence_mean": round(mean_conf, 4),
            "confidence_median": round(median_conf, 4),
            "confidence_min": round(min_conf, 4),
            "confidence_max": round(max_conf, 4),
        },
        "outputs": {
            "annotated_video": str(annotated_video_path) if not args.no_video else None,
            "detections_csv": str(csv_path),
            "detections_json": str(json_path),
        },
        "homography": homography.get_metadata() if homography is not None else None,
    }

    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Saved Run Metadata:    {metadata_path}")

    print("\n==================================================")
    print("RUN SUMMARY")
    print("==================================================")
    print(f"Frames Processed:      {frame_idx} / {metadata.frame_count}")
    print(f"Total Detections:      {len(all_detections)}")
    print(f"Detections / Frame:    {dets_per_frame:.2f}")
    print(f"Confidence (Mean/Med): {mean_conf:.3f} / {median_conf:.3f} (Min: {min_conf:.3f}, Max: {max_conf:.3f})")
    print(f"End-to-End Time:       {pipeline_total_time:.2f}s")
    print(f"Average ms / Frame:    {avg_frame_time_ms:.1f} ms")
    print(f"Model Inference Avg:   {avg_inference_ms:.1f} ms")
    print(f"Effective Throughput:  {effective_fps:.2f} FPS")
    print("==================================================")


if __name__ == "__main__":
    main()
