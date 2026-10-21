"""AR/UI application: camera + overlay + buttons, connected to the TranslationEngine.

Run from the project root:  python -m ui.app
"""
import argparse
import time

import cv2

from translation import Prediction, TranslationEngine
from vision.camera import Camera

from .buttons import Button, ButtonBar
from .overlay import draw_fps, draw_notice, draw_prediction, draw_sentence

WINDOW_NAME = "Sign Language Translator"
PRIVACY_NOTICE = "Local processing - no video is recorded or sent"
NOTICE_SECONDS = 5.0

# Fake predictor for now: keys 1, 2, 3. Replaced by the real model in week 2.
_KEY_PREDICTIONS = {
    ord("1"): Prediction("HELLO", 0.95),
    ord("2"): Prediction("THANKS", 0.95),
    ord("3"): Prediction("YES", 0.95),
}


def keyboard_predictor(frame, key):
    """get_prediction(frame, key) -> Prediction | None. Replace it with the real pipeline."""
    return _KEY_PREDICTIONS.get(key)


def run(camera_index: int = 0, get_prediction=keyboard_predictor) -> None:
    engine = TranslationEngine()

    def toggle_voice():
        engine.voice_enabled = not engine.voice_enabled

    bar = ButtonBar([
        Button("Reset", engine.reset),
        Button("Clear", engine.clear),
        Button(lambda: "Voice ON" if engine.voice_enabled else "Voice OFF",
               toggle_voice, is_on=lambda: engine.voice_enabled),
    ])

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, bar.handle_mouse)

    last_prediction = None
    start = previous = time.monotonic()
    fps = 0.0

    try:
        with Camera(camera_index) as camera:
            while True:
                frame = camera.read()
                if frame is None:
                    print("Failed to capture frame.")
                    break
                frame = cv2.flip(frame, 1)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    engine.clear()

                prediction = get_prediction(frame, key)
                engine.update(prediction)
                if prediction is not None:
                    last_prediction = prediction

                now = time.monotonic()
                dt = now - previous
                previous = now
                if dt > 0:
                    instant = 1.0 / dt
                    fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant

                if last_prediction is not None:
                    draw_prediction(frame, last_prediction.sign,
                                    last_prediction.confidence)
                draw_sentence(frame, engine.sentence)
                draw_fps(frame, fps)
                bar.draw(frame)
                if now - start < NOTICE_SECONDS:
                    draw_notice(frame, PRIVACY_NOTICE)

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except RuntimeError as error:
        print(f"Camera error: {error}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    args = parser.parse_args()
    run(args.camera)