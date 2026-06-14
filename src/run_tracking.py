from pathlib import Path
import csv
import re


# Repository root:
# video-perception-pipeline/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Input 
DETECTION_SUMMARY_CSV = REPO_ROOT / "outputs" / "detections" / "summary.csv"

# Output
TRACKS_DIR = REPO_ROOT / "outputs" / "tracks"
TRACKED_DETECTIONS_CSV = TRACKS_DIR / "tracked_detections.csv"
TRACK_SUMMARY_CSV = TRACKS_DIR / "track_summary.csv"


# Minimum IoU required to say:
# "this detection is probably the same object as an existing track"
IOU_THRESHOLD = 0.3

# How many sampled frames a track can disappear before being closed.
# Example:
# If MAX_MISSED_FRAMES = 2, the object can be missing for 2 sampled frames
# and still continue with the same track_id if it reappears.
MAX_MISSED_FRAMES = 2


def parse_frame_index(frame_name):
    """
    Extract numeric frame index from names

    Returns:
        int
    """

    match = re.search(r"frame_(\d+)", frame_name)

    if not match:
        raise ValueError(
            f"Could not parse frame index from: {frame_name}"
        )

    return int(match.group(1))


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union between two boxes.

    Box format:
        [x1, y1, x2, y2]

    IoU = overlapping_area / union_area

    Returns:
        float between 0 and 1
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)

    intersection_area = inter_width * inter_height

    # Individual areas
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def load_detections():
    """
    Load Step 6 detections from:

        outputs/detections/summary.csv

    Returns:
        list[dict]
    """

    if not DETECTION_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Detection summary not found: {DETECTION_SUMMARY_CSV}. "
            "Run Step 6 first: python3 src/run_baseline_detection.py"
        )

    detections = []

    with open(DETECTION_SUMMARY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_index = parse_frame_index(row["frame_name"])

            detections.append({
                "video_id": row["video_id"],
                "frame_name": row["frame_name"],
                "frame_index": frame_index,
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
            })

    return detections


def group_detections_by_video_and_frame(detections):
    """
    Convert flat detections into:

        grouped[video_id][frame_index] = list[detections]
    """

    grouped = {}

    for detection in detections:
        video_id = detection["video_id"]
        frame_index = detection["frame_index"]

        grouped.setdefault(video_id, {})
        grouped[video_id].setdefault(frame_index, [])
        grouped[video_id][frame_index].append(detection)

    return grouped


def create_track(track_id, detection, frame_index):
    """
    Create a new track from one detection.
    """

    return {
        "track_id": track_id,
        "class_id": detection["class_id"],
        "class_name": detection["class_name"],
        "last_box": [
            detection["x1"],
            detection["y1"],
            detection["x2"],
            detection["y2"],
        ],
        "start_frame": frame_index,
        "end_frame": frame_index,
        "last_frame": frame_index,
        "age": 1,
        "missed_frames": 0,
        "total_detections": 1,
    }


def track_video(video_id, frames_dict):
    """
    Track detections for one video.

    Strategy:
    1. Process frames in order.
    2. For every detection, find the active track with the best IoU.
    3. Match only detections with the same class.
    4. If IoU is high enough, reuse the existing track_id.
    5. Otherwise, create a new track.
    6. Close tracks that have been missing for too long.
    """

    active_tracks = []
    finished_tracks = []
    tracked_rows = []

    next_track_id = 1

    sorted_frame_indices = sorted(frames_dict.keys())

    for frame_index in sorted_frame_indices:
        detections = frames_dict[frame_index]

        # Mark all active tracks as unmatched at the beginning of this frame
        unmatched_track_indices = set(range(len(active_tracks)))

        # Sort detections by confidence so stronger detections claim tracks first
        detections = sorted(
            detections,
            key=lambda d: d["confidence"],
            reverse=True,
        )

        for detection in detections:
            detection_box = [
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
            ]

            best_track_index = None
            best_iou = 0.0

            # Find best matching active track
            for track_index in list(unmatched_track_indices):
                track = active_tracks[track_index]

                # Simple but important:
                # do not match a person detection to a spoon track, etc.
                if detection["class_id"] != track["class_id"]:
                    continue

                iou = compute_iou(
                    detection_box,
                    track["last_box"],
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track_index = track_index

            # Case 1:
            # Existing track matched
            if (
                best_track_index is not None
                and best_iou >= IOU_THRESHOLD
            ):
                track = active_tracks[best_track_index]

                missed_before_match = track["missed_frames"]

                track["last_box"] = detection_box
                track["end_frame"] = frame_index
                track["last_frame"] = frame_index
                track["age"] += 1
                track["missed_frames"] = 0
                track["total_detections"] += 1

                unmatched_track_indices.remove(best_track_index)

                tracked_rows.append({
                    **detection,
                    "track_id": track["track_id"],
                    "track_age": track["age"],
                    "is_new_track": False,
                    "matched_iou": round(best_iou, 4),
                    "missed_frames_before_match": missed_before_match,
                })

            # Case 2:
            # No good match, so this is a new track
            else:
                track = create_track(
                    track_id=next_track_id,
                    detection=detection,
                    frame_index=frame_index,
                )

                active_tracks.append(track)

                tracked_rows.append({
                    **detection,
                    "track_id": next_track_id,
                    "track_age": 1,
                    "is_new_track": True,
                    "matched_iou": 0.0,
                    "missed_frames_before_match": 0,
                })

                next_track_id += 1

        # Increase missed count for unmatched active tracks
        still_active_tracks = []

        for track_index, track in enumerate(active_tracks):

            if track_index in unmatched_track_indices:
                track["missed_frames"] += 1

            if track["missed_frames"] > MAX_MISSED_FRAMES:
                finished_tracks.append(track)
            else:
                still_active_tracks.append(track)

        active_tracks = still_active_tracks

    # At the end, all remaining active tracks are considered finished
    finished_tracks.extend(active_tracks)

    # Add video_id to track summary rows
    for track in finished_tracks:
        track["video_id"] = video_id

    return tracked_rows, finished_tracks


def save_tracked_detections(rows):
    """
    Save all detections with assigned track_id.
    """

    with open(TRACKED_DETECTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_id",
            "frame_name",
            "frame_index",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "track_id",
            "track_age",
            "is_new_track",
            "matched_iou",
            "missed_frames_before_match",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_track_summary(tracks):
    """
    Save one row per track.

    This gives:
    - track_id
    - class
    - start frame
    - end frame
    - number of detections
    - duration in sampled frames
    """

    with open(TRACK_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_id",
            "track_id",
            "class_id",
            "class_name",
            "start_frame",
            "end_frame",
            "duration_frames",
            "total_detections",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for track in tracks:
            duration_frames = (
                track["end_frame"] - track["start_frame"] + 1
            )

            writer.writerow({
                "video_id": track["video_id"],
                "track_id": track["track_id"],
                "class_id": track["class_id"],
                "class_name": track["class_name"],
                "start_frame": track["start_frame"],
                "end_frame": track["end_frame"],
                "duration_frames": duration_frames,
                "total_detections": track["total_detections"],
            })


def main():
    """
    Main tracking pipeline.

    1. Load base detections.
    2. Group detections by video and frame.
    3. Track each video independently.
    4. Save tracked detections.
    5. Save track summary.
    """

    TRACKS_DIR.mkdir(parents=True, exist_ok=True)

    detections = load_detections()
    grouped = group_detections_by_video_and_frame(detections)

    all_tracked_rows = []
    all_finished_tracks = []

    for video_id, frames_dict in grouped.items():
        print(f"Running tracking for video {video_id}")

        tracked_rows, finished_tracks = track_video(
            video_id=video_id,
            frames_dict=frames_dict,
        )

        all_tracked_rows.extend(tracked_rows)
        all_finished_tracks.extend(finished_tracks)

    save_tracked_detections(all_tracked_rows)
    save_track_summary(all_finished_tracks)

    print(f"Tracked detections saved to: {TRACKED_DETECTIONS_CSV}")
    print(f"Track summary saved to: {TRACK_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
    