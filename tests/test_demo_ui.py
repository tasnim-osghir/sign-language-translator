import numpy as np
import pytest

from demo import NO_HAND_RESET_FRAMES
from demo_ui import RealPipeline, main
from translation import Prediction, Stabilizer


def points_with(hand=True):
    points = np.full((67, 3), np.nan, dtype=np.float32)
    if hand:
        points[21:42] = 0.5
    points[42:67] = 0.5
    return points


class FakeExtractor:
    def __init__(self, hand=True):
        self.hand = hand
        self.frames = []

    def extract(self, frame):
        self.frames.append(frame)
        return points_with(self.hand)


class FakePredictor:
    def __init__(self, result=("HELLO", 0.9)):
        self.result = result
        self.inputs = []
        self.resets = 0

    def predict(self, features):
        self.inputs.append(features)
        return self.result

    def reset(self):
        self.resets += 1


def mirror(frame):
    return frame[:, ::-1]


def make_pipeline(extractor=None, predictor=None, stabilizer=None):
    return RealPipeline(
        extractor or FakeExtractor(),
        predictor or FakePredictor(),
        stabilizer or Stabilizer(window=3, min_votes=2),
        unmirror=mirror,
    )


def test_extractor_gets_the_original_frame():
    original = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    extractor = FakeExtractor()
    pipeline = make_pipeline(extractor=extractor)
    pipeline.get_prediction(mirror(original), -1)
    np.testing.assert_array_equal(extractor.frames[0], original)


def test_predictor_gets_flat_points():
    predictor = FakePredictor()
    make_pipeline(predictor=predictor).get_prediction(np.zeros((2, 2, 3)), -1)
    assert predictor.inputs[0].shape == (67 * 3,)


def test_returns_stable_prediction_after_enough_votes():
    pipeline = make_pipeline()
    frame = np.zeros((2, 2, 3))
    assert pipeline.get_prediction(frame, -1) is None
    assert pipeline.get_prediction(frame, -1) == Prediction("HELLO", 0.9)
    assert pipeline.frames == 2


def test_invalid_model_output_counts_as_no_sign():
    pipeline = make_pipeline(predictor=FakePredictor(result=("hello", 0.9)))
    frame = np.zeros((2, 2, 3))
    assert pipeline.get_prediction(frame, -1) is None
    assert pipeline.get_prediction(frame, -1) is None


def test_on_word_resets_model_and_votes():
    predictor = FakePredictor()
    pipeline = make_pipeline(predictor=predictor)
    frame = np.zeros((2, 2, 3))
    pipeline.get_prediction(frame, -1)
    pipeline.on_word("HELLO")
    assert predictor.resets == 1
    # The vote window restarted: one vote is not enough again.
    assert pipeline.get_prediction(frame, -1) is None


def test_no_hand_resets_once_after_threshold():
    predictor = FakePredictor(result=None)
    pipeline = make_pipeline(extractor=FakeExtractor(hand=False), predictor=predictor)
    frame = np.zeros((2, 2, 3))
    for _ in range(NO_HAND_RESET_FRAMES - 1):
        pipeline.get_prediction(frame, -1)
    assert predictor.resets == 0
    pipeline.get_prediction(frame, -1)
    assert predictor.resets == 1
    pipeline.get_prediction(frame, -1)
    assert predictor.resets == 1


def test_hand_restarts_the_no_hand_count():
    extractor = FakeExtractor(hand=False)
    pipeline = make_pipeline(extractor=extractor, predictor=FakePredictor(result=None))
    frame = np.zeros((2, 2, 3))
    pipeline.get_prediction(frame, -1)
    extractor.hand = True
    pipeline.get_prediction(frame, -1)
    assert pipeline.frames_without_hand == 0


@pytest.mark.parametrize(
    "argv",
    [["--min-votes", "11"], ["--window", "0"], ["--threshold", "1.5"]],
)
def test_invalid_options_exit_before_opening_anything(argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
