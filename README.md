# Video Perception Pipeline

## Objective

Build a small video perception pipeline capable of:

- Video ingestion
- Frame sampling
- Object detection
- Temporal tracking
- Prediction export
- Visualization
- Evaluation

The goal is to process egocentric task videos and generate inspectable perception outputs that can later be reviewed by humans.

---

## Selected Videos and Strategy

For the core implementation, this project focuses on two videos:

| Video ID | Category          | Primary Object | Reason                                                                                                                     |
| -------- | ----------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 165895   | Food Preparation  | Wooden Spoon   | Good manipulation sequence with hands, tool usage, occlusion, and repeated object motion.                                  |
| 839878   | Repair / Assembly | Hairdryer      | Good repair-style sequence with tool-like object handling, re-entry, viewpoint changes, and temporal reasoning challenges. |

These two videos are selected because they are better suited for demonstrating a video perception pipeline than simpler static scenes. They contain:

- object manipulation by hands
- partial occlusions
- object re-entry after disappearing
- changing viewpoints
- tool-like objects
- temporal continuity challenges

The goal is not only to detect objects frame by frame, but also to preserve useful temporal information across the video.

---

## Repository Structure

```text
video-perception-pipeline/
├── configs/
│   └── selected_videos.yaml
├── data/
├── videos/
│   ├── 165895.mp4
│   ├── 767223.mp4
│   ├── 839878.mp4
│   └── 870855.mp4
├── videos.json
└── video_manifest.md
├── docs/
│   └── Computer Vision _ Video Perception Take-Home.pdf
├── outputs/
│   ├── evaluation/
│   └── visualizations/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Metadata Files

Dataset metadata:

```text
data/videos.json
data/video_manifest.md
```

Task description:

```text
docs/Computer Vision _ Video Perception Take-Home.pdf
```

---

## Planned Pipeline

```text
Video
  │
  ▼
Frame Sampling
  │
  ▼
Object Detection
  │
  ▼
Temporal Tracking
  │
  ▼
Prediction Export (JSONL)
  │
  ▼
Visualization
  │
  ▼
Evaluation
```

---

## Planned Outputs

```text
outputs/
├── frames/
├── predictions.jsonl
├── visualizations/
└── evaluation/
```

---

## Technology Stack

- Python 3.11+
- OpenCV
- NumPy
- Ultralytics YOLO
- ByteTrack
- Pandas
- Matplotlib
