"""Run YOLO Pose tracking and the fall-candidate baseline on videos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import cv2

from src.data.pipeline import inside, read_rows
from src.pose.baseline import (
    COCO_KEYPOINTS,
    FallBaseline,
    assess_fall_candidate,
    normalize_keypoints,
)

SKELETON_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return result or "video"


def _run_directory(
    output_root: Path, source: Path, sample_id: str | None, run_id: str | None
) -> Path:
    source_key = sample_id or source.stem
    path_hash = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:8]
    timestamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parent = output_root / f"{_safe_component(source_key)}_{path_hash}"
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / _safe_component(timestamp)
    run_dir.mkdir(exist_ok=False)
    return run_dir


def _draw_person(
    frame: Any,
    bbox: list[float],
    keypoints: list[dict[str, Any]],
    track_id: int,
    confirmed: bool,
) -> None:
    color = (0, 0, 255) if confirmed else (0, 220, 0)
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {track_id}" + (" FALL?" if confirmed else "")
    cv2.putText(
        frame, label, (x1, max(20, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
    )
    points = {
        index: point
        for index, point in enumerate(keypoints)
        if point["confidence"] >= 0.25
    }
    for left, right in SKELETON_EDGES:
        if left in points and right in points:
            p1 = points[left]
            p2 = points[right]
            cv2.line(
                frame,
                (int(round(p1["x_px"])), int(round(p1["y_px"]))),
                (int(round(p2["x_px"])), int(round(p2["y_px"]))),
                color,
                2,
            )
    for point in points.values():
        cv2.circle(
            frame,
            (int(round(point["x_px"])), int(round(point["y_px"]))),
            3,
            (255, 180, 0),
            -1,
        )


def process_video(
    source: Path,
    run_dir: Path,
    model: Any,
    *,
    model_path: str,
    confidence: float,
    image_size: int,
    tracker: str,
    keypoint_confidence: float,
    minimum_box_aspect_ratio: float,
    minimum_torso_tilt_degrees: float,
    confirm_frames: int,
    device: str | None = None,
    sample_id: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Write per-frame tracked pose JSONL and an annotated video into a fresh run folder."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {source}")

    baseline = FallBaseline(confirm_frames)
    pose_temp = run_dir / "poses.jsonl.tmp"
    video_temp = run_dir / "annotated.tmp.mp4"
    pose_path = run_dir / "poses.jsonl"
    video_path = run_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_temp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise ValueError("Cannot open MP4 video writer")

    frame_index = 0
    people_count = 0
    confirmed_count = 0
    try:
        with pose_temp.open("w", encoding="utf-8", newline="\n") as stream:
            results = model.track(
                source=str(source),
                stream=True,
                persist=False,
                tracker=tracker,
                conf=confidence,
                imgsz=image_size,
                verbose=False,
                **({"device": device} if device else {}),
            )
            for result in results:
                frame = result.orig_img
                active_track_ids: set[int] = set()
                frame_people: list[dict[str, Any]] = []
                boxes = result.boxes
                keypoint_data = result.keypoints
                if boxes is not None and keypoint_data is not None and len(boxes):
                    xyxy = boxes.xyxy.cpu().numpy()
                    raw_keypoints = keypoint_data.data.cpu().numpy()
                    box_confidences = boxes.conf.cpu().numpy()
                    ids = boxes.id
                    track_ids = (
                        ids.cpu().numpy().astype(int).tolist()
                        if ids is not None
                        else [None] * len(xyxy)
                    )
                    for bbox_values, points, box_conf, track_id in zip(
                        xyxy, raw_keypoints, box_confidences, track_ids
                    ):
                        bbox = [float(value) for value in bbox_values]
                        keypoints = normalize_keypoints(
                            points.tolist(), bbox, frame_width, frame_height
                        )
                        assessment = assess_fall_candidate(
                            bbox,
                            keypoints,
                            minimum_keypoint_confidence=keypoint_confidence,
                            minimum_box_aspect_ratio=minimum_box_aspect_ratio,
                            minimum_torso_tilt_degrees=minimum_torso_tilt_degrees,
                        )
                        if track_id is None:
                            temporal = {
                                "consecutive_candidate_frames": 0,
                                "fall_confirmed": False,
                            }
                        else:
                            active_track_ids.add(track_id)
                            temporal = baseline.update(
                                track_id, assessment["fall_candidate"]
                            )
                        person = {
                            "track_id": track_id,
                            "bbox_xyxy_px": bbox,
                            "bbox_xyxy_frame_norm": [
                                bbox[0] / frame_width,
                                bbox[1] / frame_height,
                                bbox[2] / frame_width,
                                bbox[3] / frame_height,
                            ],
                            "box_confidence": float(box_conf),
                            "keypoints": keypoints,
                            "fall_baseline": {**assessment, **temporal},
                        }
                        frame_people.append(person)
                        people_count += 1
                        confirmed_count += int(temporal["fall_confirmed"])
                        if track_id is not None:
                            _draw_person(
                                frame, bbox, keypoints, track_id,
                                temporal["fall_confirmed"],
                            )

                baseline.reset_missing(active_track_ids)
                stream.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "timestamp_s": frame_index / fps,
                            "people": frame_people,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                writer.write(frame)
                frame_index += 1
        if frame_index == 0:
            raise ValueError(f"Video contains no readable frames: {source}")
        writer.release()
        capture.release()
        pose_temp.replace(pose_path)
        video_temp.replace(video_path)
    except Exception:
        writer.release()
        capture.release()
        pose_temp.unlink(missing_ok=True)
        video_temp.unlink(missing_ok=True)
        raise

    summary = {
        "source": str(source),
        "sample_id": sample_id,
        "source_label": label,
        "frames": frame_index,
        "people_detections": people_count,
        "confirmed_fall_candidate_frames": confirmed_count,
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "model": model_path,
        "tracker": tracker,
        "outputs": {
            "poses": pose_path.name,
            "annotated_video": video_path.name,
        },
        "keypoint_names": list(COCO_KEYPOINTS),
        "fall_baseline": {
            "box_aspect_ratio_min": minimum_box_aspect_ratio,
            "torso_tilt_degrees_min": minimum_torso_tilt_degrees,
            "keypoint_confidence_min": keypoint_confidence,
            "consecutive_frames": confirm_frames,
            "warning": "Heuristic candidate only; not a trained or validated classifier.",
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _sources(
    root: Path,
    source: str | None,
    manifest: str | None,
    split: str | None,
) -> list[dict[str, Any]]:
    if source:
        path = Path(source)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Video file does not exist: {path}")
        return [{"path": path, "sample_id": None, "label": None}]

    manifest_path = inside(root, manifest or "")
    rows = read_rows(manifest_path)
    selected = [row for row in rows if split is None or row.get("split") == split]
    if not selected:
        raise ValueError("Manifest selection is empty")
    sources = []
    for row in selected:
        if not row.get("processed_path") or not row.get("sample_id"):
            raise ValueError("Manifest rows need sample_id and processed_path")
        sources.append(
            {
                "path": inside(root, row["processed_path"]),
                "sample_id": str(row["sample_id"]),
                "label": row.get("label"),
            }
        )
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", help="Single video file, relative to --root or absolute")
    inputs.add_argument(
        "--manifest",
        help="Issue #1 JSONL manifest with processed_path entries",
    )
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose"))
    parser.add_argument("--run-id", help="Optional run folder name; it must not already exist")
    parser.add_argument("--config", type=Path, default=Path("configs/pose.json"))
    parser.add_argument("--model", help="Override the pose model path/name")
    parser.add_argument("--device", help="Optional Ultralytics device, e.g. cpu or 0")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_root = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        sources = _sources(root, args.source, args.manifest, args.split)
        from ultralytics import YOLO

        model_path = args.model or config["model"]
        model = YOLO(model_path)
        for item in sources:
            run_dir = _run_directory(
                output_root, item["path"], item["sample_id"], args.run_id
            )
            summary = process_video(
                item["path"],
                run_dir,
                model,
                model_path=model_path,
                confidence=config["confidence"],
                image_size=config["image_size"],
                tracker=config["tracker"],
                keypoint_confidence=config["keypoint_confidence"],
                minimum_box_aspect_ratio=config["fall_box_aspect_ratio"],
                minimum_torso_tilt_degrees=config["fall_torso_tilt_degrees"],
                confirm_frames=config["fall_confirm_frames"],
                device=args.device,
                sample_id=item["sample_id"],
                label=item["label"],
            )
            print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False))
    except (KeyError, OSError, ValueError) as exc:
        print(f"Pose pipeline error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
