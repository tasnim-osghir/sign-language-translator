
"""Detect left and right hands using MediaPipe Hands."""

import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(
        self,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        input_is_mirrored=False,
    ):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This implementation requires MediaPipe's "
                "mp.solutions API. Check your installed version."
            )

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.input_is_mirrored = input_is_mirrored

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _empty_hand():
        """Return 21 missing landmarks."""
        return np.full((21, 3), np.nan, dtype=np.float32)

    @staticmethod
    def _to_array(hand_landmarks):
        """Keep raw normalized x, y, z values."""
        return np.array(
            [
                [point.x, point.y, point.z]
                for point in hand_landmarks.landmark
            ],
            dtype=np.float32,
        )

    def process(self, frame):
        """
        Process an ORIGINAL, unflipped BGR webcam frame.

        Returns:
            {
                "left_hand": ndarray (21, 3),
                "right_hand": ndarray (21, 3),
                "results": MediaPipe results
            }
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        rgb_frame = __import__("cv2").cvtColor(
            frame, __import__("cv2").COLOR_BGR2RGB
        )

        results = self.hands.process(rgb_frame)

        left_hand = self._empty_hand()
        right_hand = self._empty_hand()

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                label = handedness.classification[0].label

                # MediaPipe Hands assumes mirrored selfie input.
                # For an original, unmirrored frame, swap the labels.
                if not self.input_is_mirrored:
                    label = (
                        "Right" if label == "Left" else "Left"
                    )

                points = self._to_array(landmarks)

                if label == "Left":
                    left_hand = points
                elif label == "Right":
                    right_hand = points

        return {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "results": results,
        }

    def draw(self, frame, results):
        """Draw detected hand landmarks on a display frame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
        return frame

    def close(self):
        self.hands.close()
