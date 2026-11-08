"""Webcam demo of the real pipeline, shown through the AR/UI of ui/.

camera -> landmarks (original frame) -> model -> stabilizer -> engine,
displayed with the overlay and buttons of ui.app (mirrored copy only).
Keys: c = clear the sentence, q or Esc = quit.
"""
import argparse
import sys

from demo import (
    DEFAULT_CAMERA,
    DEFAULT_MIN_VOTES,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    NO_HAND_RESET_FRAMES,
    allowed_only,
    hands_visible,
    to_prediction,
)
from translation import Stabilizer, TranslationEngine
from voice import Speaker

KEYBOARD_HINT = "Run 'python demo.py --keyboard' to try the demo without the models."


class RealPipeline:
    """The get_prediction / on_word pair that ui.app.run() expects.

    ui.app.run() mirrors each frame before calling get_prediction();
    unmirror undoes it (a horizontal flip is its own inverse), so the
    extractor always gets the ORIGINAL camera frame.
    allowlist: if given, a model word outside it counts as no sign, so it
    is never voted on nor shown.
    speaker: if given, each validated word is spoken (see voice.Speaker).
    """

    def __init__(self, extractor, predictor, stabilizer, unmirror, allowlist=None,
                 speaker=None):
        self._extractor = extractor
        self._predictor = predictor
        self._stabilizer = stabilizer
        self._unmirror = unmirror
        self._allowlist = allowlist
        self._speaker = speaker
        self.frames = 0
        self.frames_without_hand = 0
        self.last_points = None

    def get_prediction(self, frame, key):
        """Return the stable Prediction for this frame, or None. key is unused."""
        self.frames += 1
        points = self._extractor.extract(self._unmirror(frame))
        self.last_points = points
        left, right = hands_visible(points)

        prediction = to_prediction(self._predictor.predict(points.reshape(-1)))
        if self._allowlist is not None:
            prediction = allowed_only(prediction, self._allowlist)
        stable = self._stabilizer.update(prediction)

        if left or right:
            self.frames_without_hand = 0
        else:
            self.frames_without_hand += 1
            if self.frames_without_hand == NO_HAND_RESET_FRAMES:
                self._predictor.reset()
                self._stabilizer.reset()
        return stable

    def on_word(self, word):
        """A word was validated: the model and the votes restart."""
        self._predictor.reset()
        self._stabilizer.reset()
        if self._speaker is not None:
            # Speaks display_word(word); does nothing when the voice is off.
            self._speaker.speak(word)


def run_ui(camera_index: int, stabilizer: Stabilizer, engine: TranslationEngine) -> int:
    """Run the real pipeline in the UI window. Returns the process exit code."""
    # Imported here so that tests can import demo_ui.py without these packages.
    try:
        import cv2

        from models import predictor
        from ui import app
        from vision.landmarks import LandmarkExtractor
    except ImportError as error:
        print(f"Missing package: {error}")
        print(KEYBOARD_HINT)
        return 1

    try:
        # Loads the model now (with its hash check), not at the first sign.
        # Only the model words are allowed to reach the engine.
        engine.allowlist = predictor.words()
        extractor = LandmarkExtractor()
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"Cannot load the models: {error}")
        print(KEYBOARD_HINT)
        return 1

    speaker = Speaker(engine)
    pipeline = RealPipeline(
        extractor,
        predictor,
        stabilizer,
        unmirror=lambda frame: cv2.flip(frame, 1),
        allowlist=engine.allowlist,
        speaker=speaker,
    )
    try:
        predictor.reset()
        app.run(
            camera_index,
            get_prediction=pipeline.get_prediction,
            engine=engine,
            on_word=pipeline.on_word,
            get_landmarks=lambda: pipeline.last_points,
        )
    finally:
        speaker.close()
        extractor.close()

    # ui.app.run() prints camera errors itself but does not report them:
    # no frame processed means the camera could not be used.
    return 0 if pipeline.frames > 0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help="camera index")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="minimum confidence for a prediction to count",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help="frames in the vote window"
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=DEFAULT_MIN_VOTES,
        help="votes needed in the window to accept a sign",
    )
    args = parser.parse_args(argv)

    try:
        stabilizer = Stabilizer(
            args.window, args.min_votes, min_confidence=args.threshold
        )
        engine = TranslationEngine(threshold=args.threshold)
    except ValueError as error:
        parser.error(str(error))
    return run_ui(args.camera, stabilizer, engine)


if __name__ == "__main__":
    sys.exit(main())
