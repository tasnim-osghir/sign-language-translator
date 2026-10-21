"""Clickable buttons drawn on the OpenCV window."""
from collections.abc import Callable
from dataclasses import dataclass

import cv2

from .overlay import FONT, WHITE


@dataclass
class Button:
    label: str | Callable[[], str]
    on_click: Callable[[], None]
    is_on: Callable[[], bool] | None = None  # None = neutral button
    rect: tuple = (0, 0, 0, 0)  # x1, y1, x2, y2 (set by ButtonBar)

    def text(self) -> str:
        return self.label() if callable(self.label) else self.label

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self, frame) -> None:
        x1, y1, x2, y2 = self.rect
        if self.is_on is None:
            color = (90, 70, 50)
        elif self.is_on():
            color = (60, 140, 60)
        else:
            color = (70, 70, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 1)
        text = self.text()
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        cv2.putText(frame, text,
                    (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)


class ButtonBar:
    """A row of buttons, top-right of the window."""

    def __init__(self, buttons, width=100, height=34, gap=8, margin=8):
        self.buttons = list(buttons)
        self.width, self.height = width, height
        self.gap, self.margin = gap, margin

    def _layout(self, frame_width: int) -> None:
        total = len(self.buttons) * self.width + (len(self.buttons) - 1) * self.gap
        x = frame_width - self.margin - total
        for button in self.buttons:
            button.rect = (x, self.margin, x + self.width, self.margin + self.height)
            x += self.width + self.gap

    def draw(self, frame):
        self._layout(frame.shape[1])
        for button in self.buttons:
            button.draw(frame)
        return frame

    def handle_mouse(self, event, x, y, flags, param) -> None:
        """Pass this to cv2.setMouseCallback."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.contains(x, y):
                button.on_click()
                break