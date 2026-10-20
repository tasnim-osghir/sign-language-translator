from types import SimpleNamespace

import numpy as np
import pytest

from vision import landmarks
from vision.landmarks import LEFT, N_POINTS, POSE, RIGHT, build_points


def fake(n, value):
    """n faux landmarks dont x, y, z valent tous value."""
    return [SimpleNamespace(x=value, y=value, z=value) for _ in range(n)]


LEFT_HAND = fake(21, 0.1)
RIGHT_HAND = fake(21, 0.2)
POSE_33 = fake(33, 0.3)


def is_nan(points, part):
    return bool(np.isnan(points[part]).all())


def test_all_present():
    points = build_points(LEFT_HAND, RIGHT_HAND, POSE_33)
    assert points.shape == (N_POINTS, 3) == (67, 3)
    assert points.dtype == np.float32
    assert not np.isnan(points).any()
    assert np.allclose(points[0:21], 0.1)
    assert np.allclose(points[21:42], 0.2)
    assert np.allclose(points[42:67], 0.3)


def test_pose_keeps_first_25_points():
    pose = [SimpleNamespace(x=i / 100, y=0.5, z=0.0) for i in range(33)]
    points = build_points(None, None, pose)
    assert np.allclose(points[POSE, 0], np.arange(25) / 100)


def test_only_left():
    points = build_points(LEFT_HAND, None, None)
    assert np.allclose(points[LEFT], 0.1)
    assert is_nan(points, RIGHT)
    assert is_nan(points, POSE)


def test_only_right():
    points = build_points(None, RIGHT_HAND, [])
    assert is_nan(points, LEFT)
    assert np.allclose(points[RIGHT], 0.2)
    assert is_nan(points, POSE)


@pytest.mark.parametrize("empty", [None, []])
def test_none_detected(empty):
    points = build_points(empty, empty, empty)
    assert points.shape == (67, 3)
    assert points.dtype == np.float32
    assert np.isnan(points).all()


@pytest.mark.parametrize(
    "left, right, pose, missing",
    [
        (fake(20, 0.1), RIGHT_HAND, POSE_33, LEFT),
        (LEFT_HAND, fake(22, 0.2), POSE_33, RIGHT),
        (LEFT_HAND, RIGHT_HAND, fake(25, 0.3), POSE),
    ],
)
def test_wrong_length_counts_as_not_detected(left, right, pose, missing):
    points = build_points(left, right, pose)
    assert is_nan(points, missing)
    # Les autres parties restent remplies.
    assert int(np.isnan(points[:, 0]).sum()) == len(range(N_POINTS)[missing])


def test_swap_hands(monkeypatch):
    monkeypatch.setattr(landmarks, "SWAP_HANDS", True)
    points = build_points(LEFT_HAND, None, POSE_33)
    assert is_nan(points, LEFT)
    assert np.allclose(points[RIGHT], 0.1)
    assert np.allclose(points[POSE], 0.3)


def test_swap_hands_is_off_by_default():
    assert landmarks.SWAP_HANDS is False


def test_extractor_missing_model_raises(tmp_path):
    # Leve avant tout import de mediapipe : fonctionne sans mediapipe.
    with pytest.raises(FileNotFoundError, match="not found"):
        landmarks.LandmarkExtractor(tmp_path / "missing.task")
