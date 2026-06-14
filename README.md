# Video Perception Pipeline

## Objective

Build a small video perception pipeline capable of:

- Video ingestion
- Frame sampling
- Baseline object detection
- Temporal tracking
- Prediction export
- Visualization
- Evaluation

The goal is to process egocentric task videos and generate inspectable perception outputs that can later be reviewed by humans.

---

# Selected Videos and Strategy

For the core implementation, this project focuses on two videos:

| Video ID | Category          | Primary Object | Reason                                                                                                                     |
| -------- | ----------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 165895   | Food Preparation  | Wooden Spoon   | Good manipulation sequence with hands, tool usage, occlusion, and repeated object motion.                                  |
| 839878   | Repair / Assembly | Hairdryer      | Good repair-style sequence with tool-like object handling, re-entry, viewpoint changes, and temporal reasoning challenges. |

These two videos are selected because they are better suited for demonstrating a video perception pipeline than simpler static scenes.

They contain:

- object manipulation by hands
- partial occlusions
- object re-entry after disappearing
- changing viewpoints
- tool-like objects
- temporal continuity challenges

The goal is not only to detect objects frame by frame, but also to preserve useful temporal information across the video.

---

# Repository Structure

```text
video-perception-pipeline/
├── configs/
│   └── selected_videos.yaml
├── data/
│   ├── videos/
│   │   ├── 165895.mp4
│   │   ├── 767223.mp4
│   │   ├── 839878.mp4
│   │   └── 870855.mp4
│   ├── videos.json
│   └── video_manifest.md
├── docs/
│   └── Computer Vision _ Video Perception Take-Home.pdf
├── outputs/
│   ├── frames/
│   │   ├── 165895/
│   │   └── 839878/
│   ├── detections/
│   │   ├── 165895/
│   │   ├── 839878/
│   │   └── summary.csv
│   ├── tracks/
│   │   ├── tracked_detections.csv
│   │   └── track_summary.csv
│   ├── visualizations/
│   │   └── detections/
│   │       ├── 165895/
│   │       └── 839878/
│   └── evaluation/
├── src/
│   ├── download_videos.py
│   ├── sample_frames.py
│   ├── run_baseline_detection.py
│   └── run_tracking.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Pipeline Overview

## Step 1 — Video Download and Validation

Download the videos defined in `videos.json` and verify that OpenCV can open them correctly.

Run:

```bash
python3 src/download_videos.py
```

Validation output includes:

```text
video_id
path
fps
frame_count
duration
```

---

## Step 2 — Frame Sampling

Sample frames from the selected videos at a fixed interval.

Current configuration:

```python
SAMPLE_EVERY_SECONDS = 1.0
```

Frames are saved under:

```text
outputs/frames/<video_id>/
```

Example:

```text
outputs/frames/165895/frame_000120.jpg
outputs/frames/839878/frame_000450.jpg
```

Run:

```bash
python3 src/sample_frames.py
```

---

## Step 3 — Baseline Object Detection

Object detection is performed using Ultralytics YOLOv8.

Model:

```text
yolov8n.pt
```

Detected objects are exported as:

```text
outputs/detections/<video_id>/frame_xxxxxx.txt
```

Annotated images are exported as:

```text
outputs/visualizations/detections/<video_id>/
```

A global detection summary is generated:

```text
outputs/detections/summary.csv
```

Run:

```bash
python3 src/run_baseline_detection.py
```

---

## Step 4 — Temporal Association / Tracking

After running baseline detection, detections are associated across sampled frames using a simple IoU-based tracker.

The tracker reads:

```text
outputs/detections/summary.csv
```

and writes:

```text
outputs/tracks/tracked_detections.csv
outputs/tracks/track_summary.csv
```

Each detection receives:

```text
track_id
track_age
is_new_track
matched_iou
missed_frames_before_match
```

Each completed track contains:

```text
track_id
class_name
start_frame
end_frame
duration_frames
total_detections
```

The tracker performs:

1. Detection matching using IoU
2. Class-consistent association
3. Track creation
4. Track continuation
5. Short occlusion handling
6. Track termination

Run:

```bash
python3 src/run_tracking.py
```

This step introduces temporal reasoning by converting independent frame detections into persistent object tracks.

---

# Outputs

## Detection Outputs

```text
outputs/detections/
```

Contains:

```text
per-frame detection files
summary.csv
```

---

## Tracking Outputs

```text
outputs/tracks/
```

Contains:

```text
tracked_detections.csv
track_summary.csv
```

---

## Visualization Outputs

```text
outputs/visualizations/detections/
```

Contains:

```text
annotated detection images
```

---

## Evaluation Outputs

```text
outputs/evaluation/
```

Reserved for:

```text
manual review subset
metrics
failure analysis
reviewer feedback
```

---

# Dependencies

Install:

```bash
pip install -r requirements.txt
```

Current dependencies:

```text
opencv-python
numpy
pandas
tqdm
pyyaml
requests
ultralytics
```

---

# Full Pipeline Execution

Run the complete pipeline in order:

```bash
python3 src/download_videos.py

python3 src/sample_frames.py

python3 src/run_baseline_detection.py

python3 src/run_tracking.py
```

