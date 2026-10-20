"""Hand landmarks (21 points per hand) with MediaPipe.

Supports both MediaPipe APIs:
  - legacy  mp.solutions.hands
  - new     mp.tasks (needs models/hand_landmarker.task)

Handedness: MediaPipe assumes a mirrored (selfie) image. We feed the
ORIGINAL image, so its "Left"/"Right" label may be the opposite of the
real hand. Use SWAP_HANDEDNESS (or env VISION_SWAP_HANDEDNESS=1) to
invert. It MUST match the convention used to build the training data.
"""



import numpy as np
import os

SWAP_HANDEDNESS = os.environ.get("VISION_SWAP_HANDEDNESS", "0") == "1"
HAND_MODEL_PATH = os.environ.get(
    "VISION_HAND_MODEL", os.path.join("models", "hand_landmarker.task")
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _slot(label, swap=SWAP_HANDEDNESS):
    """Return 0 for left slot, 1 for right slot."""
    is_left = label == "Left"
    if swap:
        is_left = not is_left
    return 0 if is_left else 1


class HandTracker:
    def __init__(self, max_hands=2, det_conf=0.5, track_conf=0.5, swap=None):
        import mediapipe as mp

        self.swap = SWAP_HANDEDNESS if swap is None else swap
        self._mp = mp
        self._legacy = hasattr(mp, "solutions") and hasattr(
            mp.solutions, "hands"
        )
        if self._legacy:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not os.path.exists(HAND_MODEL_PATH):
                raise FileNotFoundError(
                    f"{HAND_MODEL_PATH} not found (needed by mp.tasks)"
                )
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=HAND_MODEL_PATH
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            self._hands = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, rgb, timestamp_ms=0):
        """Return (left, right), each (21, 3) or None. Input: ORIGINAL RGB."""
        found = []  # (slot, score, array)
        if self._legacy:
            res = self._hands.process(rgb)
            if res.multi_hand_landmarks:
                for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    cls = hd.classification[0]
                    arr = np.array(
                        [[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32
                    )
                    found.append((_slot(cls.label, self.swap), cls.score, arr))
        else:
            mp = self._mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._hands.detect_for_video(img, int(timestamp_ms))
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                cat = hd[0]
                arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                found.append((_slot(cat.category_name, self.swap), cat.score, arr))

        out = [None, None]
        best = [-1.0, -1.0]
        for slot, score, arr in found:  # keep best score per slot
            if score > best[slot]:
                best[slot] = score
                out[slot] = arr
        return out[0], out[1]

    def close(self):
        self._hands.close()