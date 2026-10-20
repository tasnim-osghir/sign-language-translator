"""Helpers for the landmark format expected by the model (models/predictor.py).

Format per frame: 67 points x (x, y, z) = 201 values, raw MediaPipe values.
    points  0-20 : left hand   (slot 0)
    points 21-41 : right hand  (slot 1)
    points 42-66 : first 25 points of MediaPipe Pose (nose = 42,
                   shoulders = 53 and 54)
A point that is not detected is NaN.
"""

import numpy as np

N_HAND = 21
N_POSE = 25
N_POINTS = 2 * N_HAND + N_POSE  # 67
N_VALUES = N_POINTS * 3  # 201

LEFT_HAND = slice(0, 21)
RIGHT_HAND = slice(21, 42)
POSE = slice(42, 67)


def bgr_to_rgb(frame):
    """OpenCV gives BGR, MediaPipe wants RGB."""
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def flip_for_display(frame):
    """Mirror the image. ONLY for display, never before MediaPipe."""
    import cv2

    return cv2.flip(frame, 1)


def mirror_points(points):
    """Mirror normalized x (x -> 1 - x) so points match a flipped display."""
    out = np.array(points, dtype=np.float32, copy=True)
    out[..., 0] = 1.0 - out[..., 0]
    return out


def empty_landmarks():
    """(67, 3) array full of NaN (nothing detected)."""
    return np.full((N_POINTS, 3), np.nan, dtype=np.float32)


def assemble_landmarks(left=None, right=None, pose=None):
    """Build the (67, 3) array from (21,3), (21,3) and (>=25,3) arrays.

    Any argument can be None (not detected) -> NaN for those points.
    Only the first 25 pose points are kept, in their original order.
    """
    out = empty_landmarks()
    if left is not None:
        out[LEFT_HAND] = np.asarray(left, dtype=np.float32).reshape(N_HAND, 3)
    if right is not None:
        out[RIGHT_HAND] = np.asarray(right, dtype=np.float32).reshape(N_HAND, 3)
    if pose is not None:
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape[0] < N_POSE:
            raise ValueError("pose needs at least 25 points")
        out[POSE] = pose[:N_POSE, :3]
    return out


def flatten(landmarks):
    """(67, 3) -> (201,) in x, y, z order per point."""
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (N_POINTS, 3):
        raise ValueError(f"expected (67, 3), got {landmarks.shape}")
    return landmarks.reshape(N_VALUES)


def detected_mask(landmarks):
    """Boolean (67,) mask: True where the point is detected."""
    return ~np.isnan(np.asarray(landmarks)).any(axis=1)


def nan_to_zero(landmarks):
    """Replace NaN by 0 (the model accepts both NaN and 0)."""
    return np.nan_to_num(np.asarray(landmarks, dtype=np.float32), nan=0.0)