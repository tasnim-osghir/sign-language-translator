"""Main entry point of the vision module.

Live use (for models/predictor.py):

    from vision.camera import Camera
    from vision.extract_landmarks import LandmarkExtractor

    extractor = LandmarkExtractor()
    with Camera() as cam:
        ok, frame = cam.read()              # ORIGINAL frame (not flipped)
        landmarks = extractor.extract(frame)        # (67, 3), NaN = missing
        flat = extractor.extract_flat(frame)        # (201,)

Offline use (build a dataset from recorded videos):

    python -m vision.extract_landmarks --input data/videos --output data/landmarks
    # data/videos/<LABEL>/*.mp4 -> data/landmarks/<LABEL>/*.npy  (T, 67, 3)
"""

import argparse
import time
from pathlib import Path

import numpy as np

from vision.preprocessing import assemble_landmarks, bgr_to_rgb, flatten


class LandmarkExtractor:
    def __init__(self, swap_handedness=None):
        from vision.hand_tracker import HandTracker
        from vision.pose_tracker import PoseTracker

        self.hands = HandTracker(swap=swap_handedness)
        self.pose = PoseTracker()
        self._last_ts = -1

    def _timestamp(self, timestamp_ms):
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms <= self._last_ts:  # must strictly increase
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        return timestamp_ms

    def extract(self, frame_bgr, timestamp_ms=None):
        """ORIGINAL BGR frame -> (67, 3) raw MediaPipe values, NaN if missing."""
        ts = self._timestamp(timestamp_ms)
        rgb = bgr_to_rgb(frame_bgr)
        left, right = self.hands.process(rgb, ts)
        pose = self.pose.process(rgb, ts)
        return assemble_landmarks(left, right, pose)

    def extract_flat(self, frame_bgr, timestamp_ms=None):
        return flatten(self.extract(frame_bgr, timestamp_ms))

    def close(self):
        self.hands.close()
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_video(path, extractor):
    """Return a (T, 67, 3) array for one video file."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extractor.extract(frame, timestamp_ms=idx * 1000.0 / fps))
        idx += 1
    cap.release()
    if not frames:
        return np.empty((0, 67, 3), dtype=np.float32)
    return np.stack(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default="data/videos")
    parser.add_argument("--output", default="data/landmarks")
    args = parser.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    videos = sorted(
        p for p in in_dir.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov"}
    )
    if not videos:
        print(f"No videos found in {in_dir}")
        return
    for path in videos:
        # A fresh extractor per video resets MediaPipe tracking and timestamps.
        with LandmarkExtractor() as extractor:
            arr = extract_video(path, extractor)
        dest = out_dir / path.parent.relative_to(in_dir) / (path.stem + ".npy")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, arr)
        print(f"{path} -> {dest} {arr.shape}")


if __name__ == "__main__":
    main()