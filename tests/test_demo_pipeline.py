import numpy as np
import pytest

from demo import hands_visible, to_prediction
from translation import Prediction


def test_to_prediction_none():
    assert to_prediction(None) is None


def test_to_prediction_valid():
    assert to_prediction(("HELLO", 0.87)) == Prediction("HELLO", 0.87)


def test_to_prediction_accepts_numpy_float():
    result = to_prediction(("YES", np.float32(0.5)))
    assert result == Prediction("YES", 0.5)
    assert type(result.confidence) is float


@pytest.mark.parametrize(
    "result",
    [
        ("hello", 0.9),        # sign id must be uppercase
        ("HELLO", 1.5),        # confidence out of [0, 1]
        ("HELLO", float("nan")),
        ("HELLO", "0.9"),      # wrong type
        (None, 0.9),
        ("HELLO",),            # wrong number of values
        ("HELLO", 0.9, 0.1),
        42,                    # not a pair at all
    ],
)
def test_to_prediction_invalid_returns_none(result):
    assert to_prediction(result) is None


def points_with(left=False, right=False):
    points = np.full((67, 3), np.nan, dtype=np.float32)
    if left:
        points[0:21] = 0.5
    if right:
        points[21:42] = 0.5
    # The pose does not count as a hand.
    points[42:67] = 0.5
    return points


def test_hands_visible_all_nan():
    assert hands_visible(np.full((67, 3), np.nan, dtype=np.float32)) == (False, False)


def test_hands_visible_pose_only():
    assert hands_visible(points_with()) == (False, False)


def test_hands_visible_only_left():
    assert hands_visible(points_with(left=True)) == (True, False)


def test_hands_visible_only_right():
    assert hands_visible(points_with(right=True)) == (False, True)


def test_hands_visible_both():
    assert hands_visible(points_with(left=True, right=True)) == (True, True)
