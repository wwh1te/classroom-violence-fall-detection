
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.pose.baseline import COCO_KEYPOINTS


CLASS_NAMES = ("normal", "violence", "fall")
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: label for label, idx in CLASS_TO_INDEX.items()}

PERSON_FEATURE_DIM = len(COCO_KEYPOINTS) * 3 + 7 + 3  # 61
INTERACTION_FEATURE_DIM = 9
FRAME_FEATURE_DIM = PERSON_FEATURE_DIM * 2 + INTERACTION_FEATURE_DIM  # 131


@dataclass
class PoseSample:
    sample_id: str
    video_id: str
    label: str | None
    sequence: np.ndarray
    metadata: dict[str, Any] | None = None


class PoseSequenceDataset(Dataset):
    """Video-level dataset containing two-person interaction sequences."""

    def __init__(
        self,
        samples: Iterable[PoseSample],
        sequence_length: int = 16,
    ):
        self.samples = list(samples)
        self.sequence_length = int(sequence_length)
        self.class_names = CLASS_NAMES
        self.class_to_index = CLASS_TO_INDEX
        self.index_to_class = INDEX_TO_CLASS

        for sample in self.samples:
            if sample.label not in CLASS_TO_INDEX:
                raise ValueError(
                    f"Invalid or missing label for sample {sample.sample_id}: "
                    f"{sample.label}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        sequence = np.asarray(sample.sequence, dtype=np.float32)

        if sequence.ndim != 2 or sequence.shape[1] != FRAME_FEATURE_DIM:
            raise ValueError(
                f"Expected sequence shape (T, {FRAME_FEATURE_DIM}), "
                f"got {sequence.shape}"
            )

        if self.sequence_length > 0:
            sequence = _fit_sequence_length(
                sequence, self.sequence_length
            )

        target = torch.tensor(
            CLASS_TO_INDEX[sample.label],
            dtype=torch.long,
        )

        return torch.tensor(sequence, dtype=torch.float32), target


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path}, line {line_number}: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"Expected a JSON object in {path}, line {line_number}"
            )

        records.append(record)

    return records


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        result = json.load(file)

    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return result


def _coerce_person_features(person: dict[str, Any]) -> np.ndarray:
    """Return 61 normalized features for one tracked person."""

    keypoint_lookup = {}

    for point in person.get("keypoints") or []:
        name = point.get("name")
        if isinstance(name, str):
            keypoint_lookup[name] = point

    features = []

    for name in COCO_KEYPOINTS:
        point = keypoint_lookup.get(name, {})

        x = float(point.get("x_frame_norm", 0.0))
        y = float(point.get("y_frame_norm", 0.0))
        confidence = float(point.get("confidence", 0.0))

        features.extend([x, y, confidence])

    # Only use normalized bounding-box coordinates.
    bbox = person.get("bbox_xyxy_frame_norm") or [0.0] * 4

    if len(bbox) >= 4:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)

        features.extend([
            float(person.get("box_confidence", 0.0)),
            width,
            height,
            x1,
            y1,
            x2,
            y2,
        ])
    else:
        features.extend([0.0] * 7)

    baseline = person.get("fall_baseline") or {}

    confirm_frames = 3

    candidate_streak = float(
        baseline.get("consecutive_candidate_frames", 0)
    )

    features.extend([
        float(bool(baseline.get("fall_candidate", False))),
        min(candidate_streak / confirm_frames, 1.0),
        float(bool(baseline.get("fall_confirmed", False))),
    ])

    result = np.asarray(features, dtype=np.float32)

    if result.shape != (PERSON_FEATURE_DIM,):
        raise ValueError(f"Unexpected person feature shape: {result.shape}")

    return result


def _bbox_iou(person_a: dict[str, Any], person_b: dict[str, Any]) -> float:
    box_a = person_a.get("bbox_xyxy_frame_norm") or [0.0] * 4
    box_b = person_b.get("bbox_xyxy_frame_norm") or [0.0] * 4

    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0

    ax1, ay1, ax2, ay2 = map(float, box_a[:4])
    bx1, by1, bx2, by2 = map(float, box_b[:4])

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _interaction_features(
    person_a: dict[str, Any] | None,
    person_b: dict[str, Any] | None,
) -> np.ndarray:
    """Return nine relative-position and presence features."""

    present_a = person_a is not None
    present_b = person_b is not None

    if not present_a or not present_b:
        return np.asarray([
            0.0, 0.0, 0.0,  # Relative center and distance
            0.0, 0.0,       # Width and height ratios
            0.0,            # Bounding-box IoU
            float(present_a and present_b),
            float(present_a),
            float(present_b),
        ], dtype=np.float32)

    box_a = person_a.get("bbox_xyxy_frame_norm") or [0.0] * 4
    box_b = person_b.get("bbox_xyxy_frame_norm") or [0.0] * 4

    ax1, ay1, ax2, ay2 = map(float, box_a[:4])
    bx1, by1, bx2, by2 = map(float, box_b[:4])

    center_ax = (ax1 + ax2) / 2
    center_ay = (ay1 + ay2) / 2
    center_bx = (bx1 + bx2) / 2
    center_by = (by1 + by2) / 2

    dx = center_bx - center_ax
    dy = center_by - center_ay
    distance = float(np.sqrt(dx ** 2 + dy ** 2))

    width_a = max(ax2 - ax1, 1e-6)
    height_a = max(ay2 - ay1, 1e-6)
    width_b = max(bx2 - bx1, 1e-6)
    height_b = max(by2 - by1, 1e-6)

    return np.asarray([
        dx,
        dy,
        distance,
        width_b / width_a,
        height_b / height_a,
        _bbox_iou(person_a, person_b),
        1.0,
        1.0,
        1.0,
    ], dtype=np.float32)


def _fit_sequence_length(
    sequence: np.ndarray,
    target_length: int,
) -> np.ndarray:
    if len(sequence) == target_length:
        return sequence

    if len(sequence) > target_length:
        indices = np.linspace(
            0, len(sequence) - 1, target_length
        ).astype(int)
        return sequence[indices]

    padding = np.zeros(
        (target_length - len(sequence), sequence.shape[1]),
        dtype=np.float32,
    )

    return np.concatenate([sequence, padding], axis=0)


def _select_top_tracks(
    frames: list[dict[str, Any]],
) -> list[str]:
    counts = Counter()

    for frame in frames:
        for person in frame.get("people") or []:
            track_id = person.get("track_id")
            if track_id is not None:
                counts[str(track_id)] += 1

    # Most frequently detected tracks first.
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [track_id for track_id, _ in ranked[:2]]


def _frame_feature_vector(
    frame: dict[str, Any],
    selected_tracks: list[str],
) -> np.ndarray:
    people_by_track = {}

    for person in frame.get("people") or []:
        track_id = person.get("track_id")

        if track_id is None:
            continue

        key = str(track_id)

        if key not in people_by_track:
            people_by_track[key] = person
        else:
            old_conf = float(
                people_by_track[key].get("box_confidence", 0.0)
            )
            new_conf = float(person.get("box_confidence", 0.0))

            if new_conf > old_conf:
                people_by_track[key] = person

    person_a = (
        people_by_track.get(selected_tracks[0])
        if len(selected_tracks) >= 1 else None
    )
    person_b = (
        people_by_track.get(selected_tracks[1])
        if len(selected_tracks) >= 2 else None
    )

    features_a = (
        _coerce_person_features(person_a)
        if person_a is not None
        else np.zeros(PERSON_FEATURE_DIM, dtype=np.float32)
    )

    features_b = (
        _coerce_person_features(person_b)
        if person_b is not None
        else np.zeros(PERSON_FEATURE_DIM, dtype=np.float32)
    )

    interaction = _interaction_features(person_a, person_b)

    return np.concatenate([
        features_a,
        features_b,
        interaction,
    ]).astype(np.float32)


def _infer_label(
    sample_id: str,
    *,
    manifest_lookup: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    label_default: str | None = None,
) -> str | None:

    if manifest_lookup is not None:
        row = manifest_lookup.get(sample_id)

        if row is None:
            return None

        label = row.get("label")
        return label if label in CLASS_NAMES else None

    if summary is not None:
        label = summary.get("source_label")
        if label in CLASS_NAMES:
            return label

    if label_default in CLASS_NAMES:
        return label_default

    return None


def load_pose_dataset(
    root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    split: str | None = None,
    sequence_length: int = 16,
    label_default: str | None = None,
) -> PoseSequenceDataset:

    root = Path(root)

    if sequence_length == 0:
        raise ValueError("sequence_length must be positive")

    if root.is_file() and root.name == "poses.jsonl":
        files = [root]
    elif root.is_dir():
        files = sorted(root.rglob("poses.jsonl"))
    else:
        raise FileNotFoundError(f"Pose data not found: {root}")

    manifest_lookup = None

    if manifest_path is not None:
        manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        manifest_lookup = {}

        for row in _read_jsonl(manifest_path):
            sample_id = row.get("sample_id")
            if sample_id:
                manifest_lookup[str(sample_id)] = row

    if split is not None and manifest_lookup is None:
        raise ValueError("A manifest is required when filtering by split")

    samples = []

    for pose_file in files:
        summary_path = pose_file.parent / "summary.json"
        summary = _read_json(summary_path)

        sample_id = summary.get("sample_id")

        if not sample_id:
            continue

        sample_id = str(sample_id)

        if manifest_lookup is not None:
            manifest_row = manifest_lookup.get(sample_id)

            # Do not allow unknown samples or missing split metadata.
            if manifest_row is None:
                continue

            if split is not None and manifest_row.get("split") != split:
                continue

        label = _infer_label(
            sample_id,
            manifest_lookup=manifest_lookup,
            summary=summary,
            label_default=label_default,
        )

        if label is None:
            continue

        frames = _read_jsonl(pose_file)

        if not frames:
            continue

        frames.sort(
            key=lambda frame: int(frame.get("frame_index", 0))
        )

        selected_tracks = _select_top_tracks(frames)

        if not selected_tracks:
            continue

        sequence = np.stack([
            _frame_feature_vector(frame, selected_tracks)
            for frame in frames
        ])

        sequence = _fit_sequence_length(
            sequence, sequence_length
        )

        samples.append(
            PoseSample(
                sample_id=sample_id,
                video_id=sample_id,
                label=label,
                sequence=sequence,
                metadata={
                    "source": str(pose_file),
                    "selected_track_ids": selected_tracks,
                    "original_frame_count": len(frames),
                    "label_status": (
                        manifest_lookup[sample_id].get("label_status")
                        if manifest_lookup is not None else None
                    ),
                },
            )
        )

    return PoseSequenceDataset(
        samples,
        sequence_length=sequence_length,
    )


def build_sequence_dataset(
    root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    split: str | None = None,
    sequence_length: int = 16,
    label_default: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:

    dataset = load_pose_dataset(
        root,
        manifest_path=manifest_path,
        split=split,
        sequence_length=sequence_length,
        label_default=label_default,
    )

    features = []
    labels = []
    ids = []

    for sample in dataset.samples:
        features.append(sample.sequence)
        labels.append(CLASS_TO_INDEX[sample.label])
        ids.append(sample.sample_id)

    if not features:
        return (
            np.empty(
                (0, sequence_length, FRAME_FEATURE_DIM),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
            [],
        )

    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        ids,
    )


__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_INDEX",
    "INDEX_TO_CLASS",
    "PERSON_FEATURE_DIM",
    "INTERACTION_FEATURE_DIM",
    "FRAME_FEATURE_DIM",
    "PoseSample",
    "PoseSequenceDataset",
    "build_sequence_dataset",
    "load_pose_dataset",
]
