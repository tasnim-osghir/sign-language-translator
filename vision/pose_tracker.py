
"""MediaPipe Pose detection for the sign-language translator."""

import cv2
import mediapipe as mp
import numpy as np


class PoseTracker:
    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "mp.solutions is unavailable in this MediaPipe installation. "
                "Check your MediaPipe version."
            )

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def empty_pose():
        """Return 25 missing pose landmarks as NaN."""
        return np.full((25, 3), np.nan, dtype=np.float32)

    def process(self, frame):
        """
        Process an original BGR frame without flipping it.

        Returns:
            pose:    NumPy array with shape (25, 3)
            results: MediaPipe Pose results
        """
        if frame is None:
            raise ValueError("The input frame cannot be None.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        pose_points = self.empty_pose()

        if results.pose_landmarks:
            all_points = np.asarray(
                [
                    [point.x, point.y, point.z]
                    for point in results.pose_landmarks.landmark
                ],
                dtype=np.float32,
            )

            # Keep Pose landmarks 0 through 24, in their original order.
            pose_points = all_points[:25].copy()

        return {
            "pose": pose_points,
            "results": results,
        }

    def draw(self, frame, results):
        """Draw pose landmarks for visualization."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
