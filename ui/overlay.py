"""Visual theme and drawing helpers for the AR interface (OpenCV only).

Colors are BGR (OpenCV order). The static decor (gradient, network dots,
panel frames, titles) is drawn once and cached; only the live parts are
drawn on each frame, so the interface stays fast.
"""
import math
from functools import lru_cache

import cv2
import numpy as np

WIDTH, HEIGHT = 1100, 640

# --- palette (BGR) ---
BG_TOP = (44, 26, 12)
BG_BOTTOM = (20, 10, 4)
PANEL = (52, 32, 18)
PANEL_DARK = (30, 18, 9)
CYAN = (235, 225, 70)
CYAN_DIM = (140, 130, 40)
WHITE = (245, 245, 245)
DIM = (165, 150, 125)
GREEN = (130, 230, 110)
ORANGE = (60, 170, 255)
RED = (100, 100, 245)

FONT = cv2.FONT_HERSHEY_DUPLEX
CONFIDENCE_OK = 0.8
FPS_TARGET = 15

# --- layout (x1, y1, x2, y2) ---
LEFT_CARD = (30, 85, 600, 565)
RIGHT_CARD = (630, 85, 1070, 565)
CAM_RECT = (50, 130, 580, 527)  # 530 x 397 (4:3)
TEXT_RECT = (650, 135, 1050, 445)
TEXT_LEFT, TEXT_RIGHT = 668, 1032
MAX_LINES = 7
LINE_HEIGHT = 38


def mix(base, other, amount):
    """Blend two BGR colors (amount 0 = base, 1 = other)."""
    return tuple(int(base[i] * (1 - amount) + other[i] * amount) for i in range(3))


def text_width(text, scale=0.6, thickness=1):
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def put(canvas, text, org, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(canvas, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def put_right(canvas, text, right, y, scale=0.6, color=WHITE, thickness=1):
    x = right - text_width(text, scale, thickness)
    put(canvas, text, (x, y), scale, color, thickness)


def rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1):
    """Rounded rectangle (thickness -1 fills it)."""
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    corners = (
        ((x1 + r, y1 + r), 180),
        ((x2 - r, y1 + r), 270),
        ((x2 - r, y2 - r), 0),
        ((x1 + r, y2 - r), 90),
    )
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for center, _ in corners:
            cv2.circle(img, center, r, color, -1, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    for center, start in corners:
        cv2.ellipse(
            img, center, (r, r), 0, start, start + 90, color, thickness, cv2.LINE_AA
        )


@lru_cache(maxsize=1)
def _static_background():
    """Gradient + network dots + glowing cards + fixed titles (built once)."""
    ramp = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32).reshape(HEIGHT, 1, 1)
    top = np.array(BG_TOP, dtype=np.float32)
    bottom = np.array(BG_BOTTOM, dtype=np.float32)
    column = top + (bottom - top) * ramp
    canvas = np.repeat(column, WIDTH, axis=1).astype(np.uint8)

    # network of dots and thin lines
    rng = np.random.default_rng(7)
    raw = rng.integers(0, [WIDTH, HEIGHT], size=(80, 2))
    points = [(int(x), int(y)) for x, y in raw]
    for i, p in enumerate(points):
        for q in points[i + 1:]:
            if math.hypot(p[0] - q[0], p[1] - q[1]) < 130:
                cv2.line(canvas, p, q, mix(BG_TOP, CYAN, 0.18), 1, cv2.LINE_AA)
    for p in points:
        cv2.circle(canvas, p, 2, mix(BG_TOP, CYAN, 0.45), -1, cv2.LINE_AA)

    # translucent cards with a glowing border
    glow_layer = np.zeros_like(canvas)
    tint = np.full_like(canvas, PANEL)
    for x1, y1, x2, y2 in (LEFT_CARD, RIGHT_CARD):
        mask = np.zeros(canvas.shape[:2], np.uint8)
        rounded_rect(mask, x1, y1, x2, y2, 22, 255, -1)
        blended = cv2.addWeighted(canvas, 0.35, tint, 0.65, 0)
        canvas[mask > 0] = blended[mask > 0]
        rounded_rect(glow_layer, x1, y1, x2, y2, 22, CYAN, 3)
    canvas = cv2.add(canvas, cv2.GaussianBlur(glow_layer, (0, 0), 9))
    for x1, y1, x2, y2 in (LEFT_CARD, RIGHT_CARD):
        rounded_rect(canvas, x1, y1, x2, y2, 22, CYAN, 2)

    # header
    title = "SIGN LANGUAGE TRANSLATOR"
    put(canvas, title, (30, 52), 0.95, WHITE)
    tag_x = 30 + text_width(title, 0.95) + 16
    rounded_rect(canvas, tag_x, 30, tag_x + 74, 62, 16, CYAN, 1)
    put(canvas, "AI + AR", (tag_x + 10, 52), 0.5, CYAN)

    privacy = "LOCAL ONLY - NO VIDEO RECORDED"
    box_right = 960
    box_left = box_right - text_width(privacy, 0.45) - 44
    rounded_rect(canvas, box_left, 32, box_right, 62, 15, PANEL_DARK, -1)
    rounded_rect(canvas, box_left, 32, box_right, 62, 15, CYAN_DIM, 1)
    cv2.circle(canvas, (box_left + 18, 47), 4, GREEN, -1, cv2.LINE_AA)
    put(canvas, privacy, (box_left + 32, 52), 0.45, DIM)

    # card titles and separators
    put(canvas, "CAMERA INPUT", (50, 112), 0.6, WHITE)
    put(canvas, "LIVE TEXT TRANSLATION", (650, 112), 0.6, WHITE)
    cv2.line(canvas, (50, 120), (580, 120), CYAN_DIM, 1, cv2.LINE_AA)
    cv2.line(canvas, (650, 120), (1050, 120), CYAN_DIM, 1, cv2.LINE_AA)
    put(canvas, "HAND GESTURE RECOGNITION", (50, 551), 0.5, DIM)

    # inset text box
    rounded_rect(canvas, *TEXT_RECT, 14, PANEL_DARK, -1)
    rounded_rect(canvas, *TEXT_RECT, 14, CYAN_DIM, 1)
    return canvas


def new_canvas():
    """A fresh copy of the static background, ready to draw the live parts on."""
    return _static_background().copy()


def wrap_words(words, max_width, scale=0.85, thickness=1):
    """Group words into lines that fit max_width pixels."""
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and text_width(candidate, scale, thickness) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_fps(canvas, fps):
    color = GREEN if fps >= FPS_TARGET else RED
    put_right(canvas, f"FPS {fps:.0f}", 1070, 52, 0.55, color)


def draw_status(canvas, text, color):
    put_right(canvas, text, 1050, 112, 0.5, color)


def draw_camera(canvas, frame, seconds, scanning=True):
    """Camera image in its card, with AR corner brackets and a scan line."""
    x1, y1, x2, y2 = CAM_RECT
    canvas[y1:y2, x1:x2] = cv2.resize(frame, (x2 - x1, y2 - y1))
    if scanning:
        y = y1 + int(seconds * 110) % (y2 - y1)
        band = canvas[y:y + 3, x1:x2]
        cv2.addWeighted(band, 0.45, np.full_like(band, CYAN), 0.55, 0, dst=band)
    arm, pad = 28, 8
    corners = ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1))
    for cx, cy, dx, dy in corners:
        px, py = cx + dx * pad, cy + dy * pad
        cv2.line(canvas, (px, py), (px + dx * arm, py), CYAN, 3, cv2.LINE_AA)
        cv2.line(canvas, (px, py), (px, py + dy * arm), CYAN, 3, cv2.LINE_AA)


def draw_sign_chip(canvas, sign, confidence):
    """AR chip on the camera image: the recognized sign and its confidence."""
    x1, _, _, y2 = CAM_RECT
    text = f"{sign.replace('_', ' ')}  {confidence * 100:.0f}%"
    left, bottom = x1 + 16, y2 - 16
    right = left + text_width(text, 0.7) + 36
    color = GREEN if confidence >= CONFIDENCE_OK else ORANGE
    rounded_rect(canvas, left, bottom - 40, right, bottom, 20, PANEL_DARK, -1)
    rounded_rect(canvas, left, bottom - 40, right, bottom, 20, color, 2)
    put(canvas, text, (left + 18, bottom - 12), 0.7, WHITE)


def draw_sentence(canvas, words):
    """The translated sentence, newest line highlighted in cyan."""
    first = TEXT_RECT[1] + 48
    if not words:
        put(canvas, "Make a sign...", (TEXT_LEFT, first), 0.8, DIM)
        return
    shown = [word.replace("_", " ") for word in words]
    lines = wrap_words(shown, TEXT_RIGHT - TEXT_LEFT)[-MAX_LINES:]
    for index, line in enumerate(lines):
        baseline = first + index * LINE_HEIGHT
        if index == len(lines) - 1:
            put(canvas, line, (TEXT_LEFT, baseline), 0.85, CYAN)
            end = TEXT_LEFT + text_width(line, 0.85)
            under = baseline + 9
            cv2.line(canvas, (TEXT_LEFT, under), (end, under), CYAN, 2, cv2.LINE_AA)
        else:
            put(canvas, line, (TEXT_LEFT, baseline), 0.8, mix(WHITE, PANEL_DARK, 0.3))


def draw_confidence(canvas, sign, confidence):
    """'Detected' label and confidence bar under the text box."""
    left, right, label_y = 650, 1050, 478
    put(canvas, "DETECTED", (left, label_y), 0.45, DIM)
    name = "-" if sign is None else sign.replace("_", " ")
    put(canvas, name, (left + 95, label_y), 0.6, CYAN)
    rounded_rect(canvas, left, 492, right, 504, 6, PANEL_DARK, -1)
    if sign is None:
        return
    value = max(0.0, min(1.0, confidence))
    width = int((right - left) * value)
    color = GREEN if value >= CONFIDENCE_OK else ORANGE
    if width >= 12:
        rounded_rect(canvas, left, 492, left + width, 504, 6, color, -1)
    put_right(canvas, f"{value * 100:.0f}%", right, label_y, 0.5, color)


def draw_footer(canvas, n_words, voice_on):
    put(canvas, f"WORDS  {n_words}", (650, 548), 0.5, DIM)
    text, color = ("VOICE ON", GREEN) if voice_on else ("VOICE OFF", RED)
    put_right(canvas, text, 1050, 548, 0.5, color)
    