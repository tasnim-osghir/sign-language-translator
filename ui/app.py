"""AR/UI application: camera + overlay + buttons, connected to the TranslationEngine.

Run from the project root:  python -m ui.app

Keys: Space = pause/start, r = reset, c = clear, v = voice on/off, q = quit.
Fake predictor keys (until the real model is plugged in): 1, 2, 3.
"""
import argparse
import time

import cv2

from translation import Prediction, TranslationEngine
from vision.camera import Camera

from .buttons import Button, ButtonBar
from .overlay import (
    CYAN,
    DIM,
    ORANGE,
    WIDTH,
    draw_camera,
    draw_confidence,
    draw_footer,
    draw_fps,
    draw_sentence,
    draw_sign_chip,
    draw_status,
    new_canvas,
)

WINDOW_NAME = "Sign Language Translator"
RECENT_SECONDS = 2.0  # how long a prediction counts as "translating"

# Fake predictor for now: keys 1, 2, 3. Replaced by the real model in week 2.
_KEY_PREDICTIONS = {
    ord("1"): Prediction("HELLO", 0.95),
    ord("2"): Prediction("THANKS", 0.95),
    ord("3"): Prediction("YES", 0.95),
}


def keyboard_predictor(frame, key):
    """get_prediction(frame, key) -> Prediction | None. Replace it with the real pipeline."""
    return _KEY_PREDICTIONS.get(key)


def run(camera_index: int = 0, get_prediction=keyboard_predictor,
        engine=None, on_word=None) -> None:
    """Open the window.

    engine: an existing TranslationEngine (created here if None).
    on_word: called with each newly validated word (e.g. by the voice module).
    """
    if engine is None:
        engine = TranslationEngine()

    state = {"paused": False, "running": True, "last": None, "last_time": -1e9}

    def toggle_voice():
        engine.voice_enabled = not engine.voice_enabled

    def toggle_pause():
        state["paused"] = not state["paused"]

    def do_clear():
        engine.clear()
        state["last"] = None

    def do_reset():
        engine.reset()
        state["last"] = None

    def quit_app():
        state["running"] = False

    bar = ButtonBar(
        [
            Button(lambda: "START" if state["paused"] else "PAUSE", toggle_pause,
                   icon=lambda: "play" if state["paused"] else "pause"),
            Button("RESET", do_reset, icon="reset"),
            Button("CLEAR", do_clear, icon="clear"),
            Button(lambda: "VOICE ON" if engine.voice_enabled else "VOICE OFF",
                   toggle_voice, is_on=lambda: engine.voice_enabled,
                   icon=lambda: "voice_on" if engine.voice_enabled else "voice_off"),
            Button("QUIT", quit_app, icon="quit", danger=True),
        ],
        canvas_width=WIDTH,
        y=578,
    )

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, bar.handle_mouse)

    start = previous = time.monotonic()
    fps = 0.0

    try:
        with Camera(camera_index) as camera:
            while state["running"]:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("Failed to capture frame.")
                    break
                frame = cv2.flip(frame, 1)
                now = time.monotonic()

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    do_clear()
                elif key == ord("r"):
                    do_reset()
                elif key == ord("v"):
                    toggle_voice()
                elif key == 32:
                    toggle_pause()

                if not state["paused"]:
                    prediction = get_prediction(frame, key)
                    word = engine.update(prediction)
                    if word is not None and on_word is not None:
                        on_word(word)
                    if prediction is not None:
                        state["last"] = prediction
                        state["last_time"] = now

                dt = now - previous
                previous = now
                if dt > 0:
                    instant = 1.0 / dt
                    fps = instant if fps == 0 else 0.9 * fps + 0.1 * instant

                if state["paused"]:
                    status, color = "PAUSED", ORANGE
                elif now - state["last_time"] < RECENT_SECONDS:
                    status, color = "TRANSLATING...", CYAN
                else:
                    status, color = "WAITING FOR SIGN...", DIM

                last = state["last"]
                sign = last.sign if last is not None else None
                confidence = last.confidence if last is not None else 0.0

                canvas = new_canvas()
                draw_camera(canvas, frame, now - start, scanning=not state["paused"])
                if last is not None:
                    draw_sign_chip(canvas, sign, confidence)
                draw_status(canvas, status, color)
                draw_sentence(canvas, engine.words)
                draw_confidence(canvas, sign, confidence)
                draw_footer(canvas, len(engine.words), engine.voice_enabled)
                draw_fps(canvas, fps)
                bar.draw(canvas)

                cv2.imshow(WINDOW_NAME, canvas)
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