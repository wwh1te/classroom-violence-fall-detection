import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.pose.pipeline import _run_directory, process_video


class ArrayValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeModel:
    def __init__(self, frames):
        self.frames = frames

    def track(self, **kwargs):
        assert kwargs["stream"] is True
        assert kwargs["persist"] is False
        return iter(self.frames)


class FakeBoxes:
    def __init__(self):
        self.xyxy = ArrayValue([[10, 8, 50, 33]])
        self.conf = ArrayValue([0.95])
        self.id = ArrayValue([7])

    def __len__(self):
        return 1


def make_result(frame):
    points = np.zeros((1, 17, 3), dtype=np.float32)
    points[:, :, 2] = 0.9
    points[0, 5] = [20, 10, 0.9]
    points[0, 6] = [20, 10, 0.9]
    points[0, 11] = [40, 20, 0.9]
    points[0, 12] = [40, 20, 0.9]
    return SimpleNamespace(
        orig_img=frame,
        boxes=FakeBoxes(),
        keypoints=SimpleNamespace(data=ArrayValue(points)),
    )


def make_video(path: Path):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48)
    )
    assert writer.isOpened()
    frames = []
    for index in range(4):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        frames.append(frame)
        writer.write(frame)
    writer.release()
    return frames


def test_process_video_exports_pose_and_temporal_fall_candidates(tmp_path):
    source = tmp_path / "source.mp4"
    frames = make_video(source)
    before = source.read_bytes()
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    model = FakeModel([make_result(frame.copy()) for frame in frames])

    summary = process_video(
        source,
        run_dir,
        model,
        model_path="fake-pose.pt",
        confidence=0.25,
        image_size=64,
        tracker="bytetrack.yaml",
        keypoint_confidence=0.25,
        minimum_box_aspect_ratio=1.25,
        minimum_torso_tilt_degrees=50,
        confirm_frames=3,
        sample_id="sample-1",
        label="fall",
    )

    records = [
        json.loads(line)
        for line in (run_dir / "poses.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["frames"] == 4
    assert summary["people_detections"] == 4
    assert [r["people"][0]["fall_baseline"]["fall_confirmed"] for r in records] == [
        False, False, True, True
    ]
    person = records[0]["people"][0]
    assert person["track_id"] == 7
    assert person["bbox_xyxy_frame_norm"] == pytest.approx(
        [10 / 64, 8 / 48, 50 / 64, 33 / 48]
    )
    assert person["keypoints"][5]["x_bbox_norm"] == pytest.approx(0.25)
    assert (run_dir / "annotated.mp4").is_file()
    assert source.read_bytes() == before


def test_run_directory_never_reuses_existing_run(tmp_path):
    source = tmp_path / "video.mp4"
    source.touch()
    first = _run_directory(tmp_path / "out", source, "sample-1", "fixed-run")

    with pytest.raises(FileExistsError):
        _run_directory(tmp_path / "out", source, "sample-1", "fixed-run")
    assert first.is_dir()
