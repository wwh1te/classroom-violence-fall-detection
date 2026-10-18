import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.pose.pipeline import FallBaseline, discover_videos, normalize_detection, process_video


def keypoints(shoulder=(30, 50), hip=(70, 55)):
    points = [[50, 30] for _ in range(17)]
    points[5] = [shoulder[0] - 5, shoulder[1]]
    points[6] = [shoulder[0] + 5, shoulder[1]]
    points[11] = [hip[0] - 5, hip[1]]
    points[12] = [hip[0] + 5, hip[1]]
    return points


def test_keypoints_and_boxes_are_normalized_to_image_dimensions():
    box, points = normalize_detection(
        [10, 20, 90, 180], keypoints(), [0.9] * 17, 100, 200)
    assert box == [0.1, 0.1, 0.9, 0.9]
    assert points[5] == dict(name="left_shoulder", x=0.25, y=0.25, confidence=0.9)
    assert points[11]["x"] == 0.65
    assert len(points) == 17


def test_fall_baseline_requires_consecutive_horizontal_torso_frames():
    baseline = FallBaseline(min_frames=3)
    points = keypoints()
    first = baseline.update(9, [10, 10, 90, 50], points, [0.9] * 17, 0)
    second = baseline.update(9, [10, 10, 90, 50], points, [0.9] * 17, 1)
    third = baseline.update(9, [10, 10, 90, 50], points, [0.9] * 17, 2)
    assert first["fall_candidate"] and not first["fall_detected"]
    assert not second["fall_detected"]
    assert third["fall_detected"]


def test_fall_baseline_resets_after_missing_frame_or_upright_posture():
    baseline = FallBaseline(min_frames=2)
    horizontal = keypoints()
    upright = keypoints(shoulder=(50, 30), hip=(50, 75))
    baseline.update(1, [10, 10, 90, 50], horizontal, [0.9] * 17, 0)
    gap_result = baseline.update(
        1, [10, 10, 90, 50], horizontal, [0.9] * 17, 2)
    upright_result = baseline.update(
        1, [10, 10, 40, 90], upright, [0.9] * 17, 3)
    assert gap_result["fall_streak"] == 1
    assert not upright_result["fall_candidate"]
    assert upright_result["fall_streak"] == 0


def test_fall_baseline_rejects_low_confidence_keypoints():
    baseline = FallBaseline(min_frames=1)
    result = baseline.update(1, [10, 10, 90, 50], keypoints(), [0.1] * 17, 0)
    assert not result["fall_candidate"]
    assert result["torso_angle_degrees"] is None


def test_discover_videos_requires_existing_source_and_supported_extension(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        discover_videos(tmp_path / "missing")
    unsupported = tmp_path / "video.txt"
    unsupported.write_text("not a video", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        discover_videos(unsupported)


def test_discover_videos_is_recursive_and_sorted(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.MP4").touch()
    (tmp_path / "a.mp4").touch()
    (tmp_path / "ignored.txt").touch()
    assert [path.name for path in discover_videos(tmp_path)] == ["a.mp4", "b.MP4"]


def test_process_video_writes_tracked_pose_jsonl_without_touching_source(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    original_bytes = source.read_bytes()

    points = np.asarray([keypoints()] * 1, dtype=np.float32)
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=np.asarray([[10, 10, 90, 50]], dtype=np.float32),
            conf=np.asarray([0.95], dtype=np.float32),
            id=np.asarray([7], dtype=np.float32),
        ),
        keypoints=SimpleNamespace(
            xy=points,
            conf=np.full((1, 17), 0.9, dtype=np.float32),
        ),
    )

    class FakeModel:
        def track(self, *args, **kwargs):
            assert args
            assert kwargs["persist"] is True
            assert kwargs["tracker"] == "bytetrack.yaml"
            return [result]

    output = tmp_path / "outputs" / "video"
    output.mkdir(parents=True)
    summary = process_video(
        source, output, FakeModel(), "bytetrack.yaml", 0.25, 640, None,
        2, 0.35, 35.0, 1.0)
    records = [json.loads(line) for line in
               (output / "frames.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["frame_count"] == 2
    assert records[0]["people"][0]["track_id"] == 7
    assert records[0]["people"][0]["keypoints"][5]["x"] == 0.25
    assert not records[0]["people"][0]["fall_baseline"]["fall_detected"]
    assert records[1]["people"][0]["fall_baseline"]["fall_detected"]
    assert source.read_bytes() == original_bytes
