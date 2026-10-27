"""Prediction stabilization: majority vote over the last N frames.

The model predicts frame by frame and can hesitate. The Stabilizer only returns
a sign if it has the most votes and was seen at least min_votes times in the window.
Predictions below min_confidence count as "no sign".
"""
import numbers
from collections import deque

from .prediction import Prediction


class Stabilizer:
    def __init__(self, window=10, min_votes=7, min_confidence=0.0):
        for name, value in (("window", window), ("min_votes", min_votes)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= min_votes <= window:
            raise ValueError("min_votes must be between 1 and window")
        if isinstance(min_confidence, bool) or not isinstance(
            min_confidence, numbers.Real
        ):
            raise TypeError("min_confidence must be a real number")
        # Written this way, the check also rejects NaN.
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")

        self.window = window
        self.min_votes = min_votes
        self.min_confidence = float(min_confidence)
        # One entry per frame: the Prediction, or None if nothing was predicted.
        self._entries = deque(maxlen=window)

    def update(self, prediction: Prediction | None) -> Prediction | None:
        """Add a prediction to the window. Return the stable sign, or None.

        A prediction below min_confidence is added as None (no sign).
        The stable sign is the one with the most votes (on a tie, the most
        recent); its confidence is the mean of its entries.
        """
        if prediction is not None and not isinstance(prediction, Prediction):
            raise TypeError("update() expects a Prediction or None")
        if prediction is not None and prediction.confidence < self.min_confidence:
            prediction = None
        self._entries.append(prediction)

        confidences = {}
        last_seen = {}
        for index, entry in enumerate(self._entries):
            if entry is None:
                continue
            confidences.setdefault(entry.sign, []).append(entry.confidence)
            last_seen[entry.sign] = index
        if not confidences:
            return None

        sign = max(confidences, key=lambda s: (len(confidences[s]), last_seen[s]))
        votes = confidences[sign]
        if len(votes) < self.min_votes:
            return None
        return Prediction(sign, sum(votes) / len(votes))

    def reset(self) -> None:
        """Empty the window."""
        self._entries.clear()
