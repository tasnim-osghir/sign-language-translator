
"""Detect left and right hand landmarks."""

import cv2
import mediapipe as mp
import numpy as np

# Member 1's requested mapping.
SWAP_HANDS = False


class HandTracker:
    def __init__(
        self,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "mp.solutions is unavailable. Check your MediaPipe version."
            )

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def empty_hand():
        return np.full((21, 3), np.nan, dtype=np.float32)

    @staticmethod
    def landmarks_to_array(hand_landmarks):
        return np.asarray(
            [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
            dtype=np.float32,
        )

    def process(self, frame):
        """Process an original, unflipped BGR frame."""
        if frame is None:
            raise ValueError("frame cannot be None")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        left = self.empty_hand()
        right = self.empty_hand()

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                label = handedness.classification[0].label

                if SWAP_HANDS:
                    label = "Right" if label == "Left" else "Left"

                points = self.landmarks_to_array(hand)

                if label == "Left":
                    left = points
                elif label == "Right":
                    right = points

        return {
            "left_hand": left,
            "right_hand": right,
            "results": results,
        }

    def draw(self, frame, results):
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS,
                )
        return frame

    def close(self):
        self.hands.close()
