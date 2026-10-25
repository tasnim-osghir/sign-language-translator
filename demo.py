"""Webcam demo of the sign language translator.

Default: the real pipeline (camera -> landmarks -> model -> stabilizer -> engine).
--keyboard: a fake keyboard predictor instead of the model
(1 = HELLO, 2 = THANKS, 3 = YES).
Keys in both modes: c = clear the sentence, q = quit.
"""
import argparse
import sys

from translation import Prediction, Stabilizer, TranslationEngine

WINDOW_NAME = "Sign Language Translator - Demo"

DEFAULT_CAMERA = 0
DEFAULT_THRESHOLD = 0.6
DEFAULT_WINDOW = 10
DEFAULT_MIN_VOTES = 7
# After this many frames in a row without a hand, the model and votes restart.
NO_HAND_RESET_FRAMES = 10
# Only the end of the sentence is drawn so that it fits on screen.
SENTENCE_CHARS = 60

_KEY_PREDICTIONS = {
    ord("1"): Prediction("HELLO", 0.95),
    ord("2"): Prediction("THANKS", 0.95),
    ord("3"): Prediction("YES", 0.95),
}


class KeyboardPredictor:
    """Stands in for the model: maps keyboard keys to predictions."""

    def predict(self, key: int) -> Prediction | None:
        """Return the prediction mapped to key, or None (also for -1, no key)."""
        return _KEY_PREDICTIONS.get(key)


def to_prediction(result) -> Prediction | None:
    """Turn the model output (WORD, probability) into a Prediction.

    Returns None if result is None or is not a valid (sign, confidence) pair.
    """
    if result is None:
        return None
    try:
        sign, confidence = result
        return Prediction(sign, confidence)
    except (TypeError, ValueError):
        return None


def hands_visible(points) -> tuple[bool, bool]:
    """Return (left, right): whether each hand has landmarks in a (67, 3) array.

    Rows 0-20 are the left hand and rows 21-41 the right hand; a hand that
    is not detected is all NaN.
    """
    import numpy as np

    points = np.asarray(points, dtype=np.float32)
    left = not np.isnan(points[0:21]).all()
    right = not np.isnan(points[21:42]).all()
    return bool(left), bool(right)


def _draw_text(cv2, image, text, origin, color):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def run_keyboard(camera_index: int = DEFAULT_CAMERA) -> None:
    # Imported here so that tests can import demo.py without OpenCV.
    import cv2

    from vision.camera import Camera

    engine = TranslationEngine()
    predictor = KeyboardPredictor()
    last_prediction = None

    try:
        with Camera(camera_index) as camera:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("Failed to capture frame.")
                    break

                frame = cv2.flip(frame, 1)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    engine.clear()

                prediction = predictor.predict(key)
                engine.update(prediction)
                if prediction is not None:
                    last_prediction = prediction

                if last_prediction is not None:
                    top_text = (
                        f"{last_prediction.sign} "
                        f"({last_prediction.confidence:.2f})"
                    )
                    _draw_text(cv2, frame, top_text, (10, 30), (0, 255, 0))

                height = frame.shape[0]
                _draw_text(
                    cv2, frame, engine.sentence, (10, height - 20), (255, 255, 255)
                )

                cv2.imshow(WINDOW_NAME, frame)

    except RuntimeError as error:
        print(f"Camera error: {error}")

    finally:
        cv2.destroyAllWindows()


def run_real(
    camera_index: int, stabilizer: Stabilizer, engine: TranslationEngine
) -> int:
    """Run the real pipeline on the webcam. Returns the process exit code."""
    # Imported here so that tests can import demo.py without these packages.
    import cv2

    from models import predictor
    from vision.camera import Camera
    from vision.landmarks import LandmarkExtractor

    extractor = None
    camera = None
    try:
        try:
            # Loads the model now (with its hash check), not at the first sign.
            predictor.words()
            extractor = LandmarkExtractor()
        except (RuntimeError, FileNotFoundError) as error:
            print(f"Cannot load the models: {error}")
            print("Run with --keyboard to try the demo without the models.")
            return 1

        camera = Camera(camera_index)
        try:
            camera.open()
        except RuntimeError as error:
            print(f"Camera error: {error}")
            return 1

        predictor.reset()
        frames_without_hand = 0

        while True:
            # The ORIGINAL frame goes to the extractor; never the mirrored copy.
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Failed to capture frame.")
                break

            points = extractor.extract(frame)
            left, right = hands_visible(points)

            prediction = to_prediction(predictor.predict(points.reshape(-1)))
            stable = stabilizer.update(prediction)
            word = engine.update(stable)
            if word is not None:
                predictor.reset()
                stabilizer.reset()

            if left or right:
                frames_without_hand = 0
            else:
                frames_without_hand += 1
                if frames_without_hand == NO_HAND_RESET_FRAMES:
                    predictor.reset()
                    stabilizer.reset()

            display = cv2.flip(frame, 1)
            if stable is not None:
                top_text = f"{stable.sign} ({stable.confidence:.2f})"
            else:
                top_text = "..."
            _draw_text(cv2, display, top_text, (10, 30), (0, 255, 0))
            hands_text = f"Hands: {'L' if left else '-'} {'R' if right else '-'}"
            _draw_text(cv2, display, hands_text, (10, 60), (0, 255, 255))
            height = display.shape[0]
            _draw_text(
                cv2,
                display,
                engine.sentence[-SENTENCE_CHARS:],
                (10, height - 20),
                (255, 255, 255),
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                engine.clear()

        return 0

    finally:
        if camera is not None:
            camera.release()
        if extractor is not None:
            extractor.close()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keyboard", action="store_true", help="use the keyboard instead of the model"
    )
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

    if args.keyboard:
        run_keyboard(args.camera)
        return 0

    try:
        stabilizer = Stabilizer(
            args.window, args.min_votes, min_confidence=args.threshold
        )
        engine = TranslationEngine(threshold=args.threshold)
    except ValueError as error:
        parser.error(str(error))
    return run_real(args.camera, stabilizer, engine)


if __name__ == "__main__":
    sys.exit(main())
