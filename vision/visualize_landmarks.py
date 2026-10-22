"""Live check of the output of vision/landmarks.py (LandmarkExtractor).

    python -m vision.visualize_landmarks

Landmarks are computed on the ORIGINAL frame. Only the picture is mirrored
for display (x -> 1 - x on the points).

Left/right test: raise ONLY your RIGHT hand. The label drawn on it must be
"R" (points 21-41). If it says "L", set SWAP_HANDS = True in
vision/landmarks.py. Keys: q = quit.
"""

import time

import cv2
import numpy as np

from vision.camera import Camera
from vision.landmarks import LEFT, POSE, RIGHT, LandmarkExtractor

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (0, 11), (0, 12),
]


def _draw(view, pts, connections, color):
    """Draw (n, 3) points on a mirrored view. Return pixel coords (n, 2)."""
    h, w = view.shape[:2]
    xy = pts[:, :2].copy()
    xy[:, 0] = 1.0 - xy[:, 0]  # mirror, to match the flipped display
    xy = xy * np.array([w, h], dtype=np.float32)
    ok = ~np.isnan(xy).any(axis=1)
    for a, b in connections:
        if ok[a] and ok[b]:
            pa = (int(xy[a, 0]), int(xy[a, 1]))
            pb = (int(xy[b, 0]), int(xy[b, 1]))
            cv2.line(view, pa, pb, color, 2)
    for i in np.flatnonzero(ok):
        cv2.circle(view, (int(xy[i, 0]), int(xy[i, 1])), 3, color, -1)
    return xy, ok


def main():
    prev = time.time()
    misses = 0
    try:
        with LandmarkExtractor() as extractor, Camera() as cam:
            while True:
                success, frame = cam.read()
                if not success:
                    frame = None
                if frame is None:
                    misses += 1
                    if misses > 30:
                        print("Failed to capture frame.")
                        break
                    continue
                misses = 0

                points = extractor.extract(frame)  # (67, 3)
                view = cv2.flip(frame, 1)  # display only

                xy_l, ok_l = _draw(view, points[LEFT], HAND_CONNECTIONS, (255, 0, 0))
                xy_r, ok_r = _draw(view, points[RIGHT], HAND_CONNECTIONS, (0, 0, 255))
                _draw(view, points[POSE], POSE_CONNECTIONS, (0, 255, 0))
                for name, xy, ok in (("L", xy_l, ok_l), ("R", xy_r, ok_r)):
                    if ok[0]:
                        pos = (int(xy[0, 0]), int(xy[0, 1]) - 10)
                        cv2.putText(view, name, pos, cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (255, 255, 255), 2)

                now = time.time()
                fps = 1.0 / max(now - prev, 1e-6)
                prev = now
                rows = int((~np.isnan(points).any(axis=1)).sum())
                cv2.putText(view, f"FPS {fps:.0f}  rows {rows}/67", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(view, "Raise your RIGHT hand: label must be R",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                cv2.imshow("landmarks", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
