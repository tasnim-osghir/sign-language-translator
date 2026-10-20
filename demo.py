"""Webcam demo: a fake keyboard predictor feeds the TranslationEngine.

Keys: 1 = HELLO, 2 = THANKS, 3 = YES, c = clear the sentence, q = quit.
"""
import argparse

from translation import Prediction, TranslationEngine

WINDOW_NAME = "Sign Language Translator - Demo"

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


def run(camera_index: int = 0) -> None:
    # Imported here so that tests can import demo.py without OpenCV.
    import cv2

    from vision.camera import Camera

    engine = TranslationEngine()
    predictor = KeyboardPredictor()
    last_prediction = None

    try:
        with Camera(camera_index) as camera:
            while True:
                frame = camera.read()
                if frame is None:
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
                    cv2.putText(
                        frame, top_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )

                height = frame.shape[0]
                cv2.putText(
                    frame, engine.sentence, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )

                cv2.imshow(WINDOW_NAME, frame)

    except RuntimeError as error:
        print(f"Camera error: {error}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    args = parser.parse_args()
    run(args.camera)
