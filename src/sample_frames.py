from pathlib import Path
import cv2
import yaml

# Repository root:
# video-perception-pipeline/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Configuration file containing the selected videos
CONFIG_PATH = REPO_ROOT / "configs" / "selected_videos.yaml"

# Directory containing downloaded videos
VIDEO_DIR = REPO_ROOT / "data" / "videos"

# Directory where sampled frames will be written
OUTPUT_DIR = REPO_ROOT / "outputs" / "frames"

# Sampling frequency:
# 1.0 means save one frame every second
SAMPLE_EVERY_SECONDS = 1.0


def load_selected_video_ids():
    """
    Load the selected video IDs from:

    Returns:
        list[str]
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return [
        str(video["video_id"])
        for video in config["selected_videos"]
    ]


def sample_video_frames(video_id):
    """
    Sample frames from a single video.

    For every N frames (corresponding to one second),
    save a JPEG image to:

        outputs/frames/<video_id>/
        
    Args:
        video_id (str)

    Returns:
        dict containing summary statistics
    """

    # Input video path
    video_path = VIDEO_DIR / f"{video_id}.mp4"

    # Output directory for sampled frames
    output_dir = OUTPUT_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify video exists
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # Open video with OpenCV
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video: {video_path}"
        )

    # Read video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            f"Invalid FPS for video: {video_path}"
        )

    # Number of frames corresponding to 1 second
    frame_interval = max(
        1,
        int(round(fps * SAMPLE_EVERY_SECONDS))
    )

    frame_index = 0
    saved_count = 0

    # Read frames sequentially
    while True:

        success, frame = cap.read()

        # End of video
        if not success:
            break

        # Save one frame every frame_interval
        if frame_index % frame_interval == 0:

            output_path = (
                output_dir /
                f"frame_{frame_index:06d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                frame
            )

            saved_count += 1

        frame_index += 1

    cap.release()

    # Compute video duration
    duration_sec = frame_index / fps

    return {
        "video_id": video_id,
        "fps": round(fps, 2),
        "total_frames": frame_index,
        "duration_sec": round(duration_sec, 2),
        "saved_frames": saved_count,
        "output_dir": str(output_dir),
    }


def main():
    """
    Main pipeline.

    1. Load selected video IDs
    2. Sample frames from each video
    3. Print a summary table
    """

    selected_video_ids = load_selected_video_ids()

    results = []

    for video_id in selected_video_ids:

        print(
            f"Sampling frames from video {video_id}"
        )

        result = sample_video_frames(video_id)

        results.append(result)

    print(
        "\nvideo_id,"
        "fps,"
        "total_frames,"
        "duration_sec,"
        "saved_frames,"
        "output_dir"
    )

    for result in results:

        print(
            f"{result['video_id']},"
            f"{result['fps']},"
            f"{result['total_frames']},"
            f"{result['duration_sec']},"
            f"{result['saved_frames']},"
            f"{result['output_dir']}"
        )


if __name__ == "__main__":
    main()