"""Clickable buttons with icons, hover effect and accent colors."""
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .overlay import CYAN, GREEN, PANEL_DARK, RED, WHITE, mix, put, rounded_rect


def _resolve(value):
    return value() if callable(value) else value


def _draw_icon(canvas, name, cx, cy, color):
    """Small vector icons drawn with OpenCV primitives."""
    aa = cv2.LINE_AA
    if name == "play":
        tri = np.array([[cx - 5, cy - 8], [cx - 5, cy + 8], [cx + 9, cy]], np.int32)
        cv2.fillPoly(canvas, [tri], color, aa)
    elif name == "pause":
        cv2.rectangle(canvas, (cx - 7, cy - 8), (cx - 2, cy + 8), color, -1)
        cv2.rectangle(canvas, (cx + 3, cy - 8), (cx + 8, cy + 8), color, -1)
    elif name == "reset":
        cv2.ellipse(canvas, (cx, cy), (8, 8), 0, 40, 330, color, 2, aa)
        cv2.circle(canvas, (cx + 6, cy - 6), 2, color, -1, aa)
    elif name == "clear":
        cv2.line(canvas, (cx - 7, cy - 7), (cx + 7, cy + 7), color, 2, aa)
        cv2.line(canvas, (cx + 7, cy - 7), (cx - 7, cy + 7), color, 2, aa)
    elif name in ("voice_on", "voice_off"):
        cv2.rectangle(canvas, (cx - 8, cy - 3), (cx - 3, cy + 3), color, -1)
        cone = np.array(
            [[cx - 3, cy - 3], [cx + 3, cy - 8], [cx + 3, cy + 8], [cx - 3, cy + 3]],
            np.int32,
        )
        cv2.fillPoly(canvas, [cone], color, aa)
        if name == "voice_on":
            cv2.ellipse(canvas, (cx + 3, cy), (6, 6), 0, -50, 50, color, 2, aa)
        else:
            cv2.line(canvas, (cx - 9, cy + 9), (cx + 9, cy - 9), color, 2, aa)
    elif name == "quit":
        cv2.ellipse(canvas, (cx, cy + 1), (8, 8), 0, -50, 230, color, 2, aa)
        cv2.line(canvas, (cx, cy - 9), (cx, cy), color, 2, aa)


@dataclass
class Button:
    label: str | Callable[[], str]
    on_click: Callable[[], None]
    is_on: Callable[[], bool] | None = None  # None = plain button
    icon: str | Callable[[], str] = ""
    danger: bool = False
    rect: tuple = (0, 0, 0, 0)  # x1, y1, x2, y2 (set by ButtonBar)
    hover: bool = False

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self, canvas) -> None:
        x1, y1, x2, y2 = self.rect
        active = self.is_on is not None and self.is_on()
        if self.danger:
            accent = RED
        elif self.is_on is None:
            accent = CYAN
        else:
            accent = GREEN if active else RED

        fill = PANEL_DARK
        if active:
            fill = mix(PANEL_DARK, accent, 0.28)
        if self.hover:
            fill = mix(fill, accent, 0.25)
        border = accent if self.hover or active else mix(accent, PANEL_DARK, 0.4)

        radius = (y2 - y1) // 2
        rounded_rect(canvas, x1, y1, x2, y2, radius, fill, -1)
        rounded_rect(canvas, x1, y1, x2, y2, radius, border, 2)

        cy = (y1 + y2) // 2
        _draw_icon(canvas, _resolve(self.icon), x1 + 30, cy, accent)
        text = _resolve(self.label)
        height = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0][1]
        put(canvas, text, (x1 + 54, cy + height // 2), 0.55, WHITE)


class ButtonBar:
    """A centered row of buttons."""

    def __init__(self, buttons, canvas_width, y, width=170, height=44, gap=14):
        self.buttons = list(buttons)
        total = len(self.buttons) * width + (len(self.buttons) - 1) * gap
        x = (canvas_width - total) // 2
        for button in self.buttons:
            button.rect = (x, y, x + width, y + height)
            x += width + gap

    def draw(self, canvas):
        for button in self.buttons:
            button.draw(canvas)
        return canvas

    def handle_mouse(self, event, x, y, flags, param) -> None:
        """Pass this to cv2.setMouseCallback."""
        if event == cv2.EVENT_MOUSEMOVE:
            for button in self.buttons:
                button.hover = button.contains(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            for button in self.buttons:
                if button.contains(x, y):
                    button.on_click()
                    break