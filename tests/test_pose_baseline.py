import pytest

from src.pose.baseline import (
    COCO_KEYPOINTS,
    FallBaseline,
    assess_fall_candidate,
    normalize_keypoints,
    torso_tilt_degrees,
)


def keypoints_for_tilt(tilt_degrees):
    import math

    radians = math.radians(tilt_degrees)
    dx, dy = math.sin(radians) * 40, math.cos(radians) * 40
    points = [[0.0, 0.0, 0.9] for _ in COCO_KEYPOINTS]
    points[5] = [100 - dx / 2, 100 - dy / 2, 0.9]
    points[6] = [100 - dx / 2, 100 - dy / 2, 0.9]
    points[11] = [100 + dx / 2, 100 + dy / 2, 0.9]
    points[12] = [100 + dx / 2, 100 + dy / 2, 0.9]
    return normalize_keypoints(points, [40, 40, 160, 160], 200, 200)


def test_keypoints_include_bbox_and_frame_normalization():
    points = normalize_keypoints(
        [[75, 80, 0.8]], [50, 40, 150, 140], 200, 200
    )

    assert points == [
        {
            "name": "nose",
            "x_px": 75.0,
            "y_px": 80.0,
            "x_frame_norm": 0.375,
            "y_frame_norm": 0.4,
            "x_bbox_norm": 0.25,
            "y_bbox_norm": 0.4,
            "confidence": 0.8,
        }
    ]


def test_torso_tilt_is_measured_in_pixels_not_stretched_bbox_coordinates():
    points = keypoints_for_tilt(70)
    assert torso_tilt_degrees(points, 0.25) == pytest.approx(70)
    result = assess_fall_candidate(
        [20, 50, 180, 150],
        points,
        minimum_box_aspect_ratio=1.25,
        minimum_torso_tilt_degrees=50,
    )
    assert result["fall_candidate"]
    assert result["box_aspect_ratio"] == pytest.approx(1.6)


def test_baseline_requires_consecutive_frames_for_the_same_track():
    baseline = FallBaseline(confirm_frames=3)

    assert baseline.update(4, True)["fall_confirmed"] is False
    assert baseline.update(4, True)["fall_confirmed"] is False
    assert baseline.update(9, True)["fall_confirmed"] is False
    assert baseline.update(4, True)["fall_confirmed"] is True
    baseline.reset_missing({9})
    assert baseline.update(4, True)["fall_confirmed"] is False


def test_baseline_rejects_invalid_threshold_and_degenerate_bbox():
    with pytest.raises(ValueError, match="at least 1"):
        FallBaseline(confirm_frames=0)
    with pytest.raises(ValueError, match="positive width"):
        assess_fall_candidate([10, 10, 10, 20], [])


def test_low_confidence_torso_does_not_trigger_candidate():
    points = keypoints_for_tilt(80)
    points[5]["confidence"] = 0.1
    result = assess_fall_candidate([20, 50, 180, 150], points)

    assert result["torso_tilt_degrees"] is None
    assert not result["fall_candidate"]
