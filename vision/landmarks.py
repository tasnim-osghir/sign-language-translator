
"""Landmark data structures, assembly helpers, and MediaPipe extractor."""

from pathlib import Path

import numpy as np

# Landmark layout expected by the model.
LEFT = slice(0, 21)
RIGHT = slice(21, 42)
POSE = slice(42, 67)

N_POINTS = 67
NUM_LANDMARKS = N_POINTS
NUM_COORDINATES = 3
NUM_FEATURES = NUM_LANDMARKS * NUM_COORDINATES

LEFT_HAND_SLICE = LEFT
RIGHT_HAND_SLICE = RIGHT
POSE_SLICE = POSE

SWAP_HANDS = False

DEFAULT_MODEL = (
    Path.home() / "Projects" / "asl-data" / "holistic_landmarker.task"
)
MODEL_ENV_VAR = "SIGN_HOLISTIC_MODEL"

def _extract_points(points, expected_count):
    """Extract x, y, z from MediaPipe landmarks, or return NaNs if invalid."""
    if points is None:
        return np.full((expected_count, 3), np.nan, dtype=np.float32)

    try:
        if len(points) != expected_count:
            return np.full(
                (expected_count, 3), np.nan, dtype=np.float32
            )

        result = np.asarray(
            [[point.x, point.y, point.z] for point in points],
            dtype=np.float32,
        )
    except (AttributeError, TypeError, ValueError):
        return np.full((expected_count, 3), np.nan, dtype=np.float32)

    if result.shape != (expected_count, 3):
        return np.full((expected_count, 3), np.nan, dtype=np.float32)

    return result


def build_points(left_hand, right_hand, pose):
    """
    Build a (67, 3) array from MediaPipe landmarks.

    Left hand:  indices 0-20
    Right hand: indices 21-41
    Pose:       first 25 of the 33 MediaPipe Pose landmarks

    Missing or invalid parts are filled with NaN.
    """
    left = _extract_points(left_hand, 21)
    right = _extract_points(right_hand, 21)

    # MediaPipe Pose is expected to provide all 33 landmarks.
    if pose is not None:
        try:
            valid_pose = len(pose) == 33
        except TypeError:
            valid_pose = False
    else:
        valid_pose = False

    if valid_pose:
        body = _extract_points(pose[:25], 25)
    else:
        body = np.full((25, 3), np.nan, dtype=np.float32)

    if SWAP_HANDS:
        left, right = right, left

    return np.concatenate((left, right, body), axis=0)


def _validate_or_fill(points, expected_shape, name):
    """Validate a NumPy landmark array or fill a missing part with NaNs."""
    if points is None:
        return np.full(expected_shape, np.nan, dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)

    if points.shape != expected_shape:
        raise ValueError(
            f"{name}: expected {expected_shape}, got {points.shape}"
        )

    return points.copy()


def combine_landmarks(left_hand, right_hand, pose):
    """
    Combine NumPy arrays into one (67, 3) frame.

    Each input must have shape (21, 3), (21, 3), or (25, 3).
    Missing inputs are filled with NaNs.
    """
    left = _validate_or_fill(left_hand, (21, 3), "left_hand")
    right = _validate_or_fill(right_hand, (21, 3), "right_hand")
    body = _validate_or_fill(pose, (25, 3), "pose")

    if SWAP_HANDS:
        left, right = right, left

    combined = np.concatenate((left, right, body), axis=0)

    assert combined.shape == (N_POINTS, NUM_COORDINATES)
    return combined


def flatten_landmarks(landmarks):
    """Convert one (67, 3) frame into a flat vector of 201 values."""
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.shape != (N_POINTS, NUM_COORDINATES):
        raise ValueError(
            f"Expected (67, 3), got {landmarks.shape}"
        )

    return landmarks.reshape(NUM_FEATURES)


def extract_pose_landmarks(pose_results):
    """Extract the first 25 Pose landmarks from legacy MediaPipe results."""
    if pose_results is None or pose_results.pose_landmarks is None:
        return np.full((25, 3), np.nan, dtype=np.float32)

    return _extract_points(
        pose_results.pose_landmarks.landmark[:25],
        25,
    )



class LandmarkExtractor:
    """Extract 67 landmarks from a BGR webcam frame using MediaPipe."""

    def __init__(self, model_path=None):
        import os

        configured_path = model_path or os.environ.get(MODEL_ENV_VAR)
        self.model_path = Path(
            configured_path if configured_path else DEFAULT_MODEL
        ).expanduser()

        # Check the model before importing MediaPipe.
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Download holistic_landmarker.task and set "
                "SIGN_HOLISTIC_MODEL to its path."
            )

        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        self._cv2 = cv2
        self._mp = mp

        options = vision.HolisticLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=VisionTaskRunningMode.IMAGE,
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(
            options
        )

    def extract(self, frame_bgr):
        """Return one (67, 3) float32 array from a BGR image."""
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a NumPy array")

        if (
            frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.dtype != np.uint8
        ):
            raise ValueError(
                "frame_bgr must have shape (height, width, 3) "
                "and dtype uint8"
            )

        frame_rgb = self._cv2.cvtColor(
            frame_bgr, self._cv2.COLOR_BGR2RGB
        )
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        result = self.landmarker.detect(image)

        return build_points(
            result.left_hand_landmarks,
            result.right_hand_landmarks,
            result.pose_landmarks,
        )

    def close(self):
        """Release MediaPipe resources."""
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
