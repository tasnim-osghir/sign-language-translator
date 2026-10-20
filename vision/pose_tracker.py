"""Pose landmarks with MediaPipe. We keep only the first 25 points.

Order is the MediaPipe order: nose = 0 (-> 42 in the final array),
left shoulder = 11 (-> 53), right shoulder = 12 (-> 54).
"""

import os

import numpy as np

from vision.preprocessing import N_POSE

POSE_MODEL_PATH = os.environ.get(
    "VISION_POSE_MODEL", os.path.join("models", "pose_landmarker.task")
)

# Upper-body connections among the first 25 points (for drawing only).
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (0, 11), (0, 12),
]


class PoseTracker:
    def __init__(self, det_conf=0.5, track_conf=0.5):
        import mediapipe as mp

        self._mp = mp
        self._legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
        if self._legacy:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not os.path.exists(POSE_MODEL_PATH):
                raise FileNotFoundError(
                    f"{POSE_MODEL_PATH} not found (needed by mp.tasks)"
                )
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=POSE_MODEL_PATH
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                min_pose_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            self._pose = mp_vision.PoseLandmarker.create_from_options(options)

    def process(self, rgb, timestamp_ms=0):
        """Return (25, 3) array (raw x, y, z) or None. Input: ORIGINAL RGB."""
        if self._legacy:
            res = self._pose.process(rgb)
            if not res.pose_landmarks:
                return None
            pts = res.pose_landmarks.landmark
        else:
            mp = self._mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._pose.detect_for_video(img, int(timestamp_ms))
            if not res.pose_landmarks:
                return None
            pts = res.pose_landmarks[0]
        return np.array(
            [[p.x, p.y, p.z] for p in pts[:N_POSE]], dtype=np.float32
        )

    def close(self):
        self._pose.close()