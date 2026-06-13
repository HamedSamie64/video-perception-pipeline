from pathlib import Path
import csv

import cv2
from ultralytics import YOLO


# Repository root:
# video-perception-pipeline/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Input frames produced by Step 5
FRAMES_DIR = REPO_ROOT / "outputs" / "frames"

# Text detection outputs
DETECTIONS_DIR = REPO_ROOT / "outputs" / "detections"

# Annotated image outputs
VISUALIZATION_DIR = (
    REPO_ROOT / "outputs" / "visualizations" / "detections"
)

# One global CSV summary for easy inspection
SUMMARY_CSV = DETECTIONS_DIR / "summary.csv"

# YOLO model:
MODEL_NAME = "yolov8n.pt"

# Confidence threshold.
# Lower values detect more objects but add more false positives.
CONFIDENCE_THRESHOLD = 0.25


def get_video_frame_dirs():
    """
    Return all video frame folders.
    """

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(
            f"Frame directory not found: {FRAMES_DIR}. "
            "Run Step 5 first: python3 src/sample_frames.py"
        )

    return sorted(
        path for path in FRAMES_DIR.iterdir()
        if path.is_dir()
    )


def run_detection_on_frame(model, frame_path, output_txt_path, output_image_path):
    """
    Run YOLO detection on one image.

    Saves:
    1. A text file containing detections
    2. An annotated visualization image

    Text format:

        class_id class_name confidence x1 y1 x2 y2
    """

    image = cv2.imread(str(frame_path))

    if image is None:
        raise RuntimeError(
            f"OpenCV could not read frame: {frame_path}"
        )

    # Run YOLO inference
    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    result = results[0]

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    detections = []

    with open(output_txt_path, "w", encoding="utf-8") as f:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

            f.write(
                f"{class_id} "
                f"{class_name} "
                f"{confidence:.4f} "
                f"{x1:.1f} "
                f"{y1:.1f} "
                f"{x2:.1f} "
                f"{y2:.1f}\n"
            )

    # Save annotated image
    annotated_image = result.plot()
    cv2.imwrite(str(output_image_path), annotated_image)

    return detections


def main():
    """
    Main detection pipeline.

    1. Load YOLO model
    2. Find sampled frames
    3. Run detection on each frame
    4. Save per-frame detections
    5. Save annotated images
    6. Save one CSV summary
    """

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    video_frame_dirs = get_video_frame_dirs()

    summary_rows = []

    for video_frame_dir in video_frame_dirs:

        video_id = video_frame_dir.name

        print(f"Running detection for video {video_id}")

        frame_paths = sorted(
            video_frame_dir.glob("*.jpg")
        )

        for frame_path in frame_paths:

            output_txt_path = (
                DETECTIONS_DIR /
                video_id /
                f"{frame_path.stem}.txt"
            )

            output_image_path = (
                VISUALIZATION_DIR /
                video_id /
                frame_path.name
            )

            detections = run_detection_on_frame(
                model=model,
                frame_path=frame_path,
                output_txt_path=output_txt_path,
                output_image_path=output_image_path,
            )

            for detection in detections:
                summary_rows.append({
                    "video_id": video_id,
                    "frame_name": frame_path.name,
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                    "confidence": round(detection["confidence"], 4),
                    "x1": round(detection["x1"], 1),
                    "y1": round(detection["y1"], 1),
                    "x2": round(detection["x2"], 1),
                    "y2": round(detection["y2"], 1),
                })

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_id",
            "frame_name",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Detection summary saved to: {SUMMARY_CSV}")
    print(f"Annotated frames saved to: {VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()