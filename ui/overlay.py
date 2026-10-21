"""AR overlay drawing helpers: they take a frame and text, and draw on the frame."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
GRAY = (170, 170, 170)

CONFIDENCE_OK = 0.8
FPS_TARGET = 15


def _dim_rect(frame, x1, y1, x2, y2, alpha=0.55):
    """Darken a rectangle of the frame so the text on it stays readable."""
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0, dst=roi)


def _fit_text(text, max_width, scale, thickness):
    """Keep the end of the text (the newest words) if it is too wide."""
    def width_of(value):
        return cv2.getTextSize(value, FONT, scale, thickness)[0][0]

    if width_of(text) <= max_width:
        return text
    while len(text) > 1 and width_of("..." + text) > max_width:
        text = text[1:]
    return "..." + text


def draw_prediction(frame, sign, confidence):
    """Top-left panel: current predicted sign and its confidence."""
    if sign is None or confidence is None:
        return frame
    text = f"{sign} ({confidence:.2f})"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    _dim_rect(frame, 8, 8, 8 + tw + 20, 8 + th + 20)
    color = GREEN if confidence >= CONFIDENCE_OK else ORANGE
    cv2.putText(frame, text, (18, 18 + th), FONT, 0.8, color, 2, cv2.LINE_AA)
    return frame


def draw_sentence(frame, sentence):
    """Bottom bar: the validated sentence (subtitle style)."""
    height, width = frame.shape[:2]
    _dim_rect(frame, 0, height - 50, width, height, alpha=0.6)
    if sentence:
        text = _fit_text(sentence, width - 24, 0.8, 2)
        color = WHITE
    else:
        text = "Make a sign..."
        color = GRAY
    cv2.putText(frame, text, (12, height - 18), FONT, 0.8, color, 2, cv2.LINE_AA)
    return frame


def draw_fps(frame, fps):
    """FPS counter (green if the 15 FPS target is met, red otherwise)."""
    height, width = frame.shape[:2]
    color = GREEN if fps >= FPS_TARGET else RED
    cv2.putText(frame, f"{fps:.0f} FPS", (width - 90, height - 60),
                FONT, 0.55, color, 1, cv2.LINE_AA)
    return frame


def draw_notice(frame, text):
    """Centered banner (used for the privacy message and for 'No hand')."""
    height, width = frame.shape[:2]
    scale = 0.6
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x = max(0, (width - tw) // 2)
    y = height // 2
    _dim_rect(frame, x - 12, y - th - 12, x + tw + 12, y + 12, alpha=0.7)
    cv2.putText(frame, text, (x, y), FONT, scale, WHITE, 1, cv2.LINE_AA)
    return frame