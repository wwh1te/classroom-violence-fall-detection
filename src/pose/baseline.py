"""Geometry-based fall-candidate baseline; this is not a trained classifier."""
from __future__ import annotations

import math
from typing import Any

COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def normalize_keypoints(
    keypoints: list[list[float]],
    bbox_xyxy: list[float],
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Return COCO keypoints with pixel, frame-relative, and bbox-relative coordinates."""
    x1, y1, x2, y2 = bbox_xyxy
    box_width = max(x2 - x1, 1e-6)
    box_height = max(y2 - y1, 1e-6)
    normalized: list[dict[str, Any]] = []
    for index, point in enumerate(keypoints):
        if len(point) < 2:
            raise ValueError("Each keypoint must contain x and y coordinates")
        x, y = float(point[0]), float(point[1])
        confidence = float(point[2]) if len(point) > 2 else 1.0
        name = COCO_KEYPOINTS[index] if index < len(COCO_KEYPOINTS) else f"keypoint_{index}"
        normalized.append(
            {
                "name": name,
                "x_px": x,
                "y_px": y,
                "x_frame_norm": x / frame_width,
                "y_frame_norm": y / frame_height,
                "x_bbox_norm": (x - x1) / box_width,
                "y_bbox_norm": (y - y1) / box_height,
                "confidence": confidence,
            }
        )
    return normalized


def torso_tilt_degrees(
    keypoints: list[dict[str, Any]], minimum_confidence: float
) -> float | None:
    """Measure shoulder-to-hip axis from vertical in image pixels."""
    indexed = {point["name"]: point for point in keypoints}
    torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(
        name not in indexed or indexed[name]["confidence"] < minimum_confidence
        for name in torso_names
    ):
        return None

    shoulder_x = (
        indexed["left_shoulder"]["x_px"] + indexed["right_shoulder"]["x_px"]
    ) / 2
    shoulder_y = (
        indexed["left_shoulder"]["y_px"] + indexed["right_shoulder"]["y_px"]
    ) / 2
    hip_x = (indexed["left_hip"]["x_px"] + indexed["right_hip"]["x_px"]) / 2
    hip_y = (indexed["left_hip"]["y_px"] + indexed["right_hip"]["y_px"]) / 2
    delta_x, delta_y = hip_x - shoulder_x, hip_y - shoulder_y
    if math.hypot(delta_x, delta_y) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(delta_x), abs(delta_y)))


def assess_fall_candidate(
    bbox_xyxy: list[float],
    keypoints: list[dict[str, Any]],
    *,
    minimum_keypoint_confidence: float = 0.25,
    minimum_box_aspect_ratio: float = 1.25,
    minimum_torso_tilt_degrees: float = 50.0,
) -> dict[str, Any]:
    """Apply a transparent per-frame geometry rule, not a probability estimate."""
    x1, y1, x2, y2 = bbox_xyxy
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("Bounding box must have positive width and height")
    aspect_ratio = width / height
    torso_tilt = torso_tilt_degrees(keypoints, minimum_keypoint_confidence)
    candidate = (
        aspect_ratio >= minimum_box_aspect_ratio
        and torso_tilt is not None
        and torso_tilt >= minimum_torso_tilt_degrees
    )
    return {
        "fall_candidate": candidate,
        "box_aspect_ratio": aspect_ratio,
        "torso_tilt_degrees": torso_tilt,
        "rule": "wide_box_and_tilted_torso",
    }


class FallBaseline:
    """Require a geometric fall candidate on consecutive frames of the same track."""

    def __init__(self, confirm_frames: int = 3) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        self.confirm_frames = confirm_frames
        self._streaks: dict[int, int] = {}

    def update(self, track_id: int, candidate: bool) -> dict[str, Any]:
        streak = self._streaks.get(track_id, 0) + 1 if candidate else 0
        self._streaks[track_id] = streak
        return {
            "consecutive_candidate_frames": streak,
            "fall_confirmed": streak >= self.confirm_frames,
        }

    def reset_missing(self, active_track_ids: set[int]) -> None:
        for track_id in self._streaks.keys() - active_track_ids:
            self._streaks[track_id] = 0
