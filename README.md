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
├── data/
├── docs/
├── outputs/
├── src/
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

## Baseline Object Detection

Model:

```text
yolov8n.pt
```

Run:

```bash
python3 src/run_baseline_detection.py
```

## Temporal Tracking

```bash
python3 src/run_tracking.py
```

## Prediction Export

```bash
python3 src/export_predictions_jsonl.py
```

## Prediction Visualization

```bash
python3 src/visualize_predictions.py
```

## Manual Evaluation Preparation

```bash
python3 src/prepare_manual_evaluation.py
```

## Baseline Evaluation

```bash
python3 src/evaluate_predictions.py
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

The feedback captures common perception and tracking issues observed during manual review, including:

```text
missed detections
partial visibility
occlusions
tracking instability
viewpoint changes
ID consistency issues
potential false positives
```

The purpose is to demonstrate a human-in-the-loop workflow where reviewer observations are used to guide targeted improvements to the perception pipeline.

---

## Reviewer-Guided Pipeline Improvement

A bounded improvement was implemented based on reviewer feedback.

### Improvement

Very small detection boxes are filtered before tracking and export:

```python
MIN_BOX_AREA = 10000
```

Detections with a bounding-box area below this threshold are removed. This reduces low-value detections that are more likely to correspond to clutter, partial objects, or unstable predictions.

### Evaluation Comparison

| Metric               | Baseline | Improved |
| -------------------- | -------- | -------- |
| Predictions Exported | 2236     | 2212     |
| True Positives       | 6        | 6        |
| False Positives      | 0        | 0        |
| Missed Detections    | 94       | 94       |
| Precision            | 1.00     | 1.00     |
| Recall               | 0.06     | 0.06     |

### Outcome

The improvement removed 24 small detections from the prediction export while preserving all reviewed target-object detections. Precision and recall remained unchanged, indicating that the removed detections did not contribute to the evaluated wooden spoon or hairdryer instances.

This demonstrates a simple reviewer-guided refinement process without modifying the detector model or tracking algorithm.

---

# Outputs

```text
outputs/detections/
outputs/tracks/
outputs/predictions.jsonl
outputs/visualizations/
outputs/evaluation/
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
```
