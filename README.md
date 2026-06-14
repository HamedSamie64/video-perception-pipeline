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
│   ├── videos.json
│   └── video_manifest.md
├── docs/
├── outputs/
│   ├── frames/
│   ├── detections/
│   ├── tracks/
│   ├── predictions.jsonl
│   ├── visualizations/
│   │   ├── detections/
│   │   └── predictions/
│   │       └── clips/
│   └── evaluation/
├── src/
│   ├── download_videos.py
│   ├── sample_frames.py
│   ├── run_baseline_detection.py
│   ├── run_tracking.py
│   ├── export_predictions_jsonl.py
│   └── visualize_predictions.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Pipeline Overview

## Video Download and Validation

```bash
python3 src/download_videos.py
```

Validates:

```text
video_id
path
fps
frame_count
duration
```

---

## Frame Sampling

```bash
python3 src/sample_frames.py
```

Frames are stored under:

```text
outputs/frames/<video_id>/
```

---

## Baseline Object Detection

Model:

```text
yolov8n.pt
```

Run:

```bash
python3 src/run_baseline_detection.py
```

Outputs:

```text
outputs/detections/
outputs/visualizations/detections/
```

---

## Temporal Tracking

Run:

```bash
python3 src/run_tracking.py
```

Outputs:

```text
outputs/tracks/tracked_detections.csv
outputs/tracks/track_summary.csv
```

Features:

- IoU-based association
- Class-consistent matching
- Track creation
- Track continuation
- Occlusion handling
- Track termination

---

## Prediction Export

Run:

```bash
python3 src/export_predictions_jsonl.py
```

Output:

```text
outputs/predictions.jsonl
```

Each prediction contains:

```json
{
  "video_id": "839878",
  "frame_index": 1234,
  "timestamp_sec": 41.1,
  "class_label": "hair dryer",
  "box": [x1, y1, x2, y2],
  "track_id": 3,
  "confidence": 0.72,
  "method": "YOLO + IoU Tracker",
  "notes": "possible occlusion or re-entry"
}
```

---

## Prediction Visualization

Tracked predictions are rendered back onto sampled frames for inspection.

Run:

```bash
python3 src/visualize_predictions.py
```

Reads:

```text
outputs/frames/<video_id>/
outputs/tracks/tracked_detections.csv
```

Writes:

```text
outputs/visualizations/predictions/<video_id>/
outputs/visualizations/predictions/clips/
```

Each visualization includes:

```text
bounding box
class label
confidence score
track_id
new track marker
possible re-entry marker
```

Short MP4 clips are generated to inspect temporal continuity and track stability.

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

## Prediction Outputs

```text
outputs/predictions.jsonl
```

---

## Visualization Outputs

```text
outputs/visualizations/detections/
```

Contains YOLO detection-only annotated images.

```text
outputs/visualizations/predictions/
```

Contains:

```text
bounding boxes
class labels
confidence scores
track IDs
new track markers
possible re-entry markers
```

Short clips:

```text
outputs/visualizations/predictions/clips/
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

```bash
pip install -r requirements.txt
```

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

```bash
python3 src/download_videos.py

python3 src/sample_frames.py

python3 src/run_baseline_detection.py

python3 src/run_tracking.py

python3 src/export_predictions_jsonl.py

python3 src/visualize_predictions.py
```
