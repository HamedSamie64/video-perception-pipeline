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
│   ├── evaluation/
│   │   ├── manual_labels.csv
│   │   └── reviewer_feedback.csv
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
│   ├── visualize_predictions.py
│   ├── prepare_manual_evaluation.py
│   └── evaluate_predictions.py
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

## Frame Sampling

```bash
python3 src/sample_frames.py
```

Frames are stored under:

```text
outputs/frames/<video_id>/
```

Metadata is stored in:

```text
outputs/frames/frame_metadata.csv
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

---

## Prediction Visualization

Run:

```bash
python3 src/visualize_predictions.py
```

Outputs:

```text
outputs/visualizations/predictions/
outputs/visualizations/predictions/clips/
```

---

## Manual Evaluation Preparation

Manual labels are stored in:

```text
data/evaluation/manual_labels.csv
```

Run:

```bash
python3 src/prepare_manual_evaluation.py
```

Output:

```text
outputs/evaluation/manual_vs_predictions.csv
```

This file allows side-by-side inspection of manually reviewed labels and exported predictions.

---

## Baseline Evaluation

Run:

```bash
python3 src/evaluate_predictions.py
```

Inputs:

```text
data/evaluation/manual_labels.csv
outputs/predictions.jsonl
```

Outputs:

```text
outputs/evaluation/evaluation_details.csv
outputs/evaluation/metrics_summary.csv
outputs/evaluation/tracking_metrics.csv
```

Detection metrics:

```text
true positives
false positives
missed detections
true negatives
precision
recall
```

Tracking metrics:

```text
track fragmentation
ID switches
```

---

## Simulated Reviewer Feedback

Reviewer feedback is stored in:

```text
data/evaluation/reviewer_feedback.csv
```

This file contains simulated human-review comments describing common perception and tracking failure modes observed during inspection of prediction outputs.

Example feedback categories:

```text
missed detections
partial visibility
occlusions
tracking instability
viewpoint changes
ID consistency issues
potential false positives
```

The feedback is not used as ground-truth annotations. Instead, it represents observations that a human reviewer might provide after inspecting visualizations and evaluation results.

The purpose of this step is to demonstrate a human-in-the-loop review workflow:

```text
predictions
    ↓
visual inspection
    ↓
reviewer feedback
    ↓
pipeline improvement
```

The reviewer feedback is used to guide the bounded improvement step that follows.

---

# Outputs

## Detection Outputs

```text
outputs/detections/
```

## Tracking Outputs

```text
outputs/tracks/
```

## Prediction Outputs

```text
outputs/predictions.jsonl
```

## Visualization Outputs

```text
outputs/visualizations/predictions/
```

## Evaluation Outputs

```text
outputs/evaluation/
```

Contains:

```text
manual_vs_predictions.csv
evaluation_details.csv
metrics_summary.csv
tracking_metrics.csv
```

Human review artifacts:

```text
data/evaluation/manual_labels.csv
data/evaluation/reviewer_feedback.csv
```

---

# Dependencies

```bash
pip install -r requirements.txt
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

python3 src/prepare_manual_evaluation.py

python3 src/evaluate_predictions.py

# Human review artifacts
# data/evaluation/manual_labels.csv
# data/evaluation/reviewer_feedback.csv
```
