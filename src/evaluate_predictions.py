from pathlib import Path
import csv
import json


# Repository root:
# video-perception-pipeline/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Human-reviewed object presence labels
MANUAL_LABELS_CSV = (
    REPO_ROOT / "data" / "evaluation" / "manual_labels.csv"
)

# Exported tracked predictions
PREDICTIONS_JSONL = REPO_ROOT / "outputs" / "predictions.jsonl"

# Evaluation output directory
EVALUATION_DIR = REPO_ROOT / "outputs" / "evaluation"

# Detailed per-frame evaluation result
EVALUATION_DETAILS_CSV = EVALUATION_DIR / "evaluation_details.csv"

# Main metric summary
METRICS_SUMMARY_CSV = EVALUATION_DIR / "metrics_summary.csv"

# Tracking-specific metric summary
TRACKING_METRICS_CSV = EVALUATION_DIR / "tracking_metrics.csv"


# The manual labels = human-friendly target names.
# YOLO output = slightly different class names.
# Creat this mapping for the evaluaiton.
TARGET_CLASS_ALIASES = {
    "wooden spoon": {
        "wooden spoon",
        "spoon",
    },
    "hairdryer": {
        "hairdryer",
        "hair dryer",
        "hair drier",
    },
}


def normalize_label(label):
    """
    Normalize labels so simple string differences do not break evaluation.
    """

    return label.strip().lower().replace("_", " ")


def is_target_prediction(predicted_class, manual_class):
    """
    Check whether a model prediction matches the manually reviewed target.

    Example:
        manual label: hairdryer
        YOLO label: hair drier
    """

    manual_class = normalize_label(manual_class)
    predicted_class = normalize_label(predicted_class)

    aliases = TARGET_CLASS_ALIASES.get(
        manual_class,
        {manual_class},
    )

    aliases = {
        normalize_label(alias)
        for alias in aliases
    }

    return predicted_class in aliases


def load_manual_labels():
    """
    Load human-reviewed frame-level labels.
    """

    if not MANUAL_LABELS_CSV.exists():
        raise FileNotFoundError(
            f"Manual labels not found: {MANUAL_LABELS_CSV}"
        )

    rows = []

    with open(MANUAL_LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "video_id": row["video_id"],
                "frame_name": row["frame_name"],
                "frame_index": int(row["frame_index"]),
                "class_label": row["class_label"],
                "present": int(row["present"]),
                "notes": row["notes"],
            })

    return rows


def load_predictions():
    """
    Load exported prediction JSONL.

    One line is one tracked detection.
    """

    if not PREDICTIONS_JSONL.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_JSONL}. "
            "Run: python3 src/export_predictions_jsonl.py"
        )

    predictions = []

    with open(PREDICTIONS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))

    return predictions


def group_predictions_by_frame(predictions):
    """
    Group predictions by video and original frame index.
    """

    grouped = {}

    for prediction in predictions:
        key = (
            prediction["video_id"],
            int(prediction["frame_index"]),
        )

        grouped.setdefault(key, [])
        grouped[key].append(prediction)

    return grouped


def evaluate_frames(manual_labels, predictions_by_frame):
    """
    Compare manual labels with predictions on the reviewed frames.

    This is frame-level presence evaluation:
    - true positive: target object is present and predicted
    - missed detection: target object is present but not predicted
    - false positive: target object is absent but predicted
    - true negative: target object is absent and not predicted
    """

    details = []

    counts = {
        "true_positives": 0,
        "false_positives": 0,
        "missed_detections": 0,
        "true_negatives": 0,
    }

    for label in manual_labels:
        key = (
            label["video_id"],
            label["frame_index"],
        )

        frame_predictions = predictions_by_frame.get(key, [])

        target_predictions = [
            prediction
            for prediction in frame_predictions
            if is_target_prediction(
                predicted_class=prediction["class_label"],
                manual_class=label["class_label"],
            )
        ]

        manual_present = label["present"] == 1
        predicted_present = len(target_predictions) > 0

        if manual_present and predicted_present:
            outcome = "true_positive"
            counts["true_positives"] += 1

        elif manual_present and not predicted_present:
            outcome = "missed_detection"
            counts["missed_detections"] += 1

        elif not manual_present and predicted_present:
            outcome = "false_positive"
            counts["false_positives"] += 1

        else:
            outcome = "true_negative"
            counts["true_negatives"] += 1

        predicted_classes = [
            prediction["class_label"]
            for prediction in frame_predictions
        ]

        target_track_ids = [
            str(prediction["track_id"])
            for prediction in target_predictions
        ]

        target_confidences = [
            str(round(prediction["confidence"], 3))
            for prediction in target_predictions
        ]

        details.append({
            "video_id": label["video_id"],
            "frame_name": label["frame_name"],
            "frame_index": label["frame_index"],
            "manual_class_label": label["class_label"],
            "manual_present": int(manual_present),
            "predicted_present": int(predicted_present),
            "outcome": outcome,
            "num_predictions_on_frame": len(frame_predictions),
            "num_target_predictions": len(target_predictions),
            "predicted_classes": ";".join(predicted_classes),
            "target_track_ids": ";".join(target_track_ids),
            "target_confidences": ";".join(target_confidences),
            "manual_notes": label["notes"],
        })

    return details, counts


def compute_detection_metrics(counts):
    """
    Compute precision and recall from frame-level counts.
    """

    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    missed_detections = counts["missed_detections"]

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + missed_detections

    precision = (
        true_positives / precision_denominator
        if precision_denominator > 0
        else None
    )

    recall = (
        true_positives / recall_denominator
        if recall_denominator > 0
        else None
    )

    return {
        **counts,
        "precision": precision,
        "recall": recall,
    }


def compute_tracking_metrics(evaluation_details):
    """
    Compute simple tracking quality metrics on target detections.

    Track fragmentation:
        Number of extra track IDs used for the same target object.

    ID switches:
        Number of times the target object changes track ID between
        consecutive reviewed frames where it was detected.
    """

    grouped = {}

    for row in evaluation_details:
        key = (
            row["video_id"],
            row["manual_class_label"],
        )

        grouped.setdefault(key, [])
        grouped[key].append(row)

    tracking_rows = []

    for (video_id, class_label), rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda row: row["frame_index"],
        )

        detected_track_sequence = []

        for row in rows:
            if row["target_track_ids"]:
                first_track_id = row["target_track_ids"].split(";")[0]
                detected_track_sequence.append(first_track_id)

        unique_track_ids = sorted(set(detected_track_sequence))

        track_fragmentation = max(
            0,
            len(unique_track_ids) - 1,
        )

        id_switches = 0

        for previous_id, current_id in zip(
            detected_track_sequence,
            detected_track_sequence[1:],
        ):
            if previous_id != current_id:
                id_switches += 1

        tracking_rows.append({
            "video_id": video_id,
            "class_label": class_label,
            "reviewed_target_detections": len(detected_track_sequence),
            "unique_track_ids": len(unique_track_ids),
            "track_ids": ";".join(unique_track_ids),
            "track_fragmentation": track_fragmentation,
            "id_switches": id_switches,
        })

    return tracking_rows


def save_evaluation_details(details):
    """
    Save per-frame evaluation result.
    """

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "frame_name",
        "frame_index",
        "manual_class_label",
        "manual_present",
        "predicted_present",
        "outcome",
        "num_predictions_on_frame",
        "num_target_predictions",
        "predicted_classes",
        "target_track_ids",
        "target_confidences",
        "manual_notes",
    ]

    with open(
        EVALUATION_DETAILS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)


def save_metrics_summary(metrics):
    """
    Save detection metric summary.
    """

    fieldnames = [
        "true_positives",
        "false_positives",
        "missed_detections",
        "true_negatives",
        "precision",
        "recall",
    ]

    with open(
        METRICS_SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics)


def save_tracking_metrics(tracking_rows):
    """
    Save tracking metric summary.
    """

    fieldnames = [
        "video_id",
        "class_label",
        "reviewed_target_detections",
        "unique_track_ids",
        "track_ids",
        "track_fragmentation",
        "id_switches",
    ]

    with open(
        TRACKING_METRICS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tracking_rows)


def format_metric(value):
    """
    Format metrics for terminal output.
    """

    if value is None:
        return "undefined"

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def main():
    """
    Evaluation pipeline.

    1. Load human-reviewed labels.
    2. Load exported tracked predictions.
    3. Compare target-object presence per reviewed frame.
    4. Compute frame-level detection metrics.
    5. Compute simple tracking stability metrics.
    6. Save CSV reports.
    """

    manual_labels = load_manual_labels()
    predictions = load_predictions()

    predictions_by_frame = group_predictions_by_frame(
        predictions=predictions,
    )

    details, counts = evaluate_frames(
        manual_labels=manual_labels,
        predictions_by_frame=predictions_by_frame,
    )

    detection_metrics = compute_detection_metrics(
        counts=counts,
    )

    tracking_metrics = compute_tracking_metrics(
        evaluation_details=details,
    )

    save_evaluation_details(
        details=details,
    )

    save_metrics_summary(
        metrics=detection_metrics,
    )

    save_tracking_metrics(
        tracking_rows=tracking_metrics,
    )

    print("Evaluation complete")
    print(f"Manual labels: {len(manual_labels)}")
    print(f"Predictions: {len(predictions)}")
    print("")
    print("Detection metrics")
    print(f"  true positives: {detection_metrics['true_positives']}")
    print(f"  false positives: {detection_metrics['false_positives']}")
    print(f"  missed detections: {detection_metrics['missed_detections']}")
    print(f"  true negatives: {detection_metrics['true_negatives']}")
    print(f"  precision: {format_metric(detection_metrics['precision'])}")
    print(f"  recall: {format_metric(detection_metrics['recall'])}")
    print("")
    print(f"Saved: {EVALUATION_DETAILS_CSV}")
    print(f"Saved: {METRICS_SUMMARY_CSV}")
    print(f"Saved: {TRACKING_METRICS_CSV}")


if __name__ == "__main__":
    main()