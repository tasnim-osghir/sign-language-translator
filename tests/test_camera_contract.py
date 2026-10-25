"""The callers of Camera.read() follow its (success, frame) contract.

No webcam and no window: the camera is a fake and the OpenCV GUI calls are
stubbed, but cv2.flip and the drawing run for real on the returned frame.
"""
import sys
import types

import numpy as np
import pytest

import demo
from translation import Stabilizer, TranslationEngine

# Both must stop the loop with the "Failed to capture frame." message.
FAILED_READS = [(False, None), (True, None)]


def make_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class FakeCamera:
    """Stands in for vision.camera.Camera: read() returns (success, frame)."""

    def __init__(self, reads):
        self.reads = list(reads)
        self.released = False

    def open(self):
        return True

    def read(self):
        return self.reads.pop(0)

    def release(self):
        self.released = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


class FakeExtractor:
    def __init__(self):
        self.frames = []
        self.closed = False

    def extract(self, frame):
        self.frames.append(frame)
        return np.full((67, 3), np.nan, dtype=np.float32)

    def close(self):
        self.closed = True


class FakePredictor:
    def words(self):
        return ["HELLO"]

    def reset(self):
        pass

    def predict(self, features):
        return None


class FakeSpeaker:
    """Stands in for voice.Speaker: no pyttsx3, no audio device, prints nothing."""

    def __init__(self):
        self.words = []
        self.closed = False

    def speak(self, word):
        self.words.append(word)
        return True

    def close(self):
        self.closed = True


@pytest.fixture
def shown(monkeypatch):
    """Stub the OpenCV window calls; returns the list of images shown."""
    cv2 = pytest.importorskip("cv2")
    images = []
    monkeypatch.setattr(cv2, "imshow", lambda name, image: images.append(image))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, callback: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: 1)
    return images


@pytest.mark.parametrize("failed_read", FAILED_READS)
def test_keyboard_mode_shows_frame_then_stops(monkeypatch, capsys, shown, failed_read):
    camera = FakeCamera([(True, make_frame()), failed_read])
    monkeypatch.setattr("vision.camera.Camera", lambda index: camera)

    demo.run_keyboard()

    assert len(shown) == 1
    assert capsys.readouterr().out == "Failed to capture frame.\n"
    assert camera.reads == []
    assert camera.released


@pytest.mark.parametrize("failed_read", FAILED_READS)
def test_real_mode_shows_frame_then_stops(monkeypatch, capsys, shown, failed_read):
    frame = make_frame()
    camera = FakeCamera([(True, frame), failed_read])
    extractor = FakeExtractor()
    monkeypatch.setattr("vision.camera.Camera", lambda index: camera)
    monkeypatch.setattr("vision.landmarks.LandmarkExtractor", lambda: extractor)
    models = types.ModuleType("models")
    models.predictor = FakePredictor()
    monkeypatch.setitem(sys.modules, "models", models)
    # run_real() does "from voice import Speaker" at call time.
    speaker = FakeSpeaker()
    monkeypatch.setattr("voice.Speaker", lambda engine: speaker)

    code = demo.run_real(0, Stabilizer(window=3, min_votes=2), TranslationEngine())

    assert code == 0
    assert len(shown) == 1
    # The extractor gets the frame itself, not the (success, frame) tuple.
    assert len(extractor.frames) == 1
    assert extractor.frames[0] is frame
    assert capsys.readouterr().out == "Failed to capture frame.\n"
    assert camera.released
    assert extractor.closed
    assert speaker.closed


@pytest.mark.parametrize("failed_read", FAILED_READS)
def test_ui_app_shows_frame_then_stops(monkeypatch, capsys, shown, failed_read):
    from ui import app

    camera = FakeCamera([(True, make_frame()), failed_read])
    monkeypatch.setattr(app, "Camera", lambda index: camera)

    app.run(0)

    assert len(shown) == 1
    assert capsys.readouterr().out == "Failed to capture frame.\n"
    assert camera.reads == []
    assert camera.released
