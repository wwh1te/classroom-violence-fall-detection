"""Extract tracked YOLO pose detections and a heuristic fall baseline."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import uuid

VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class FallBaseline:
    """Rule-based fall baseline using persistent horizontal torso posture."""

    def __init__(self, min_frames=5, keypoint_confidence=0.35, max_torso_angle=35.0,
                 min_box_aspect_ratio=1.0):
        if min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        if not 0 <= keypoint_confidence <= 1:
            raise ValueError("keypoint_confidence must be between 0 and 1")
        if not 0 < max_torso_angle < 90:
            raise ValueError("max_torso_angle must be between 0 and 90")
        if min_box_aspect_ratio <= 0:
            raise ValueError("min_box_aspect_ratio must be positive")
        self.min_frames = min_frames
        self.keypoint_confidence = keypoint_confidence
        self.max_torso_angle = max_torso_angle
        self.min_box_aspect_ratio = min_box_aspect_ratio
        self._streaks = defaultdict(int)
        self._last_frames = {}

    def update(self, track_id, bbox, keypoints, confidences, frame_index):
        if track_id is None:
            return dict(fall_candidate=False, fall_detected=False, fall_streak=0,
                        torso_angle_degrees=None)

        if self._last_frames.get(track_id) != frame_index - 1:
            self._streaks[track_id] = 0
        self._last_frames[track_id] = frame_index

        shoulder_indices, hip_indices = (5, 6), (11, 12)
        if (len(keypoints) < len(COCO_KEYPOINTS)
                or len(confidences) < len(COCO_KEYPOINTS)
                or any(confidences[i] < self.keypoint_confidence
                       for i in (*shoulder_indices, *hip_indices))):
            self._streaks[track_id] = 0
            return dict(fall_candidate=False, fall_detected=False, fall_streak=0,
                        torso_angle_degrees=None)

        shoulder = tuple(sum(keypoints[i][axis] for i in shoulder_indices) / 2
                         for axis in range(2))
        hip = tuple(sum(keypoints[i][axis] for i in hip_indices) / 2
                    for axis in range(2))
        dx, dy = abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1])
        torso_angle = math.degrees(math.atan2(dy, dx))
        x1, y1, x2, y2 = bbox
        box_width, box_height = x2 - x1, y2 - y1
        candidate = (
            torso_angle <= self.max_torso_angle
            and box_height > 0
            and box_width / box_height >= self.min_box_aspect_ratio
        )
        self._streaks[track_id] = self._streaks[track_id] + 1 if candidate else 0
        return dict(
            fall_candidate=candidate,
            fall_detected=self._streaks[track_id] >= self.min_frames,
            fall_streak=self._streaks[track_id],
            torso_angle_degrees=round(torso_angle, 2),
        )


def normalize_detection(bbox, keypoints, keypoint_confidences, frame_width, frame_height):
    """Serialize a detection using [0, 1] image-relative coordinates."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive")
    normalized_box = [
        min(1.0, max(0.0, float(bbox[0]) / frame_width)),
        min(1.0, max(0.0, float(bbox[1]) / frame_height)),
        min(1.0, max(0.0, float(bbox[2]) / frame_width)),
        min(1.0, max(0.0, float(bbox[3]) / frame_height)),
    ]
    points = []
    for index, name in enumerate(COCO_KEYPOINTS):
        if index < len(keypoints):
            x, y = keypoints[index]
            confidence = (float(keypoint_confidences[index])
                          if index < len(keypoint_confidences) else 0.0)
            points.append(dict(
                name=name,
                x=min(1.0, max(0.0, float(x) / frame_width)),
                y=min(1.0, max(0.0, float(y) / frame_height)),
                confidence=confidence,
            ))
        else:
            points.append(dict(name=name, x=None, y=None, confidence=0.0))
    return normalized_box, points


def discover_videos(source):
    source = Path(source)
    if not source.exists():
        raise ValueError(f"Input path does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported video extension: {source.suffix}")
        return [source.resolve()]
    videos = sorted(path.resolve() for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError(f"No video files found under: {source}")
    return videos


def _tensor_array(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _tracked_people(result, frame_width, frame_height, baseline, frame_index):
    boxes = result.boxes
    xyxy = _tensor_array(boxes.xyxy)
    box_confidences = _tensor_array(boxes.conf)
    track_ids = _tensor_array(boxes.id)
    pose_xy = _tensor_array(result.keypoints.xy)
    pose_confidences = _tensor_array(result.keypoints.conf)
    if pose_xy is None:
        return []
    if pose_confidences is None:
        pose_confidences = pose_xy[:, :, 0] * 0

    people = []
    for index, box in enumerate(xyxy):
        track_id = int(track_ids[index]) if track_ids is not None else None
        keypoints = pose_xy[index].tolist()
        confidences = pose_confidences[index].tolist()
        pixel_box = [float(value) for value in box]
        normalized_box, normalized_points = normalize_detection(
            pixel_box, keypoints, confidences, frame_width, frame_height)
        baseline_result = baseline.update(
            track_id, pixel_box, keypoints, confidences, frame_index)
        people.append(dict(
            track_id=track_id,
            bbox_xyxy_pixels=[round(value, 3) for value in pixel_box],
            bbox_xyxy_normalized=[round(value, 6) for value in normalized_box],
            detection_confidence=float(box_confidences[index]),
            keypoints=normalized_points,
            fall_baseline=baseline_result,
        ))
    return people


def _video_output_name(video, source_root):
    try:
        relative = video.relative_to(source_root)
    except ValueError:
        relative = Path(video.name)
    stem = "_".join(relative.with_suffix("").parts)
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
    digest = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:10]
    return f"{stem[:100]}-{digest}"


def process_video(video, output_dir, model, tracker, confidence, image_size, device,
                  min_fall_frames, fall_keypoint_confidence, max_torso_angle,
                  min_box_aspect_ratio):
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Invalid video FPS: {video}")

    kwargs = dict(persist=True, tracker=tracker, conf=confidence,
                  imgsz=image_size, verbose=False)
    if device:
        kwargs["device"] = device
    fall_baseline = FallBaseline(
        min_frames=min_fall_frames,
        keypoint_confidence=fall_keypoint_confidence,
        max_torso_angle=max_torso_angle,
        min_box_aspect_ratio=min_box_aspect_ratio,
    )
    frame_path = output_dir / "frames.jsonl"
    frame_count = detection_count = fall_frame_count = 0
    try:
        with frame_path.open("x", encoding="utf-8") as stream:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                results = model.track(frame, **kwargs)
                height, width = frame.shape[:2]
                people = _tracked_people(
                    results[0], width, height, fall_baseline, frame_count)
                detection_count += len(people)
                fall_frame_count += sum(
                    person["fall_baseline"]["fall_detected"] for person in people)
                record = dict(
                    frame_index=frame_count,
                    timestamp_seconds=round(frame_count / fps, 6),
                    width=width,
                    height=height,
                    people=people,
                )
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                frame_count += 1
    finally:
        capture.release()
    if frame_count == 0:
        raise ValueError(f"Video contains no readable frames: {video}")
    return dict(video=str(video), frames_file=frame_path.name, fps=fps,
                frame_count=frame_count, person_detections=detection_count,
                fall_detected_person_frames=fall_frame_count)


def run(args):
    source = Path(args.source).resolve()
    videos = discover_videos(source)
    if source.is_file():
        source_root = source.parent
    else:
        source_root = source
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir()

    run_info = dict(
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        source=str(source),
        model=args.model,
        tracker=args.tracker,
        coordinate_format="normalized image coordinates in [0, 1]; boxes also include pixel xyxy",
        keypoint_format="COCO-17 in the order documented in src/pose/pipeline.py",
        fall_baseline=dict(
            type="horizontal torso and wide bounding box sustained over consecutive frames",
            min_frames=args.min_fall_frames,
            keypoint_confidence=args.fall_keypoint_confidence,
            max_torso_angle_degrees=args.max_torso_angle,
            min_box_aspect_ratio=args.min_box_aspect_ratio,
        ),
        videos=[],
    )
    run_info_path = run_dir / "run.json"
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from ultralytics import YOLO

        pose_model = YOLO(args.model)
        for video in videos:
            name = _video_output_name(video, source_root)
            video_output = run_dir / name
            video_output.mkdir()
            summary = process_video(
                video, video_output, pose_model, args.tracker, args.confidence,
                args.image_size, args.device, args.min_fall_frames,
                args.fall_keypoint_confidence, args.max_torso_angle,
                args.min_box_aspect_ratio)
            run_info["videos"].append(dict(output_dir=name, **summary))
            run_info_path.write_text(
                json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Processed {video}: {summary['frame_count']} frames", flush=True)
        run_info["status"] = "completed"
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        run_info["status"] = "failed"
        run_info["error"] = str(exc)
        raise
    finally:
        run_info["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_info_path.write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to: {run_dir}")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed"),
                        help="Video file or directory; defaults to normalized project videos")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/pose_tracking"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--min-fall-frames", type=int, default=5)
    parser.add_argument("--fall-keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--max-torso-angle", type=float, default=35.0)
    parser.add_argument("--min-box-aspect-ratio", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not 0 < args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    try:
        run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Pose pipeline error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
