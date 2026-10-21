
"""Combine hand and pose landmarks into the model input format."""

import numpy as np

NUM_HAND_POINTS = 21
NUM_POSE_POINTS = 25
NUM_TOTAL_POINTS = 67
NUM_COORDINATES = 3


def _validate_points(points, expected_rows, name):
    if points is None:
        return np.full(
            (expected_rows, NUM_COORDINATES),
            np.nan,
            dtype=np.float32,
        )

    points = np.asarray(points, dtype=np.float32)

    if points.shape != (expected_rows, NUM_COORDINATES):
        raise ValueError(
            f"{name} must have shape "
            f"({expected_rows}, 3), got {points.shape}"
        )

    return points.copy()


def combine_landmarks(left_hand, right_hand, pose):
    """Combine 21 left-hand, 21 right-hand and 25 pose points."""
    left = _validate_points(left_hand, 21, "left_hand")
    right = _validate_points(right_hand, 21, "right_hand")
    body = _validate_points(pose, 25, "pose")

    landmarks = np.concatenate((left, right, body), axis=0)

    if landmarks.shape != (67, 3):
        raise RuntimeError(f"Unexpected shape: {landmarks.shape}")

    return landmarks


def flatten_landmarks(landmarks):
    """Convert (67, 3) landmarks into 201 values."""
    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.shape != (67, 3):
        raise ValueError(
            f"Expected shape (67, 3), got {landmarks.shape}"
        )

    return landmarks.reshape(201)


def landmarks_to_model_input(landmarks):
    """Prepare one frame for the model."""
    return flatten_landmarks(landmarks)
