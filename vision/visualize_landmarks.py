"""Live visualisation to check the landmarks and the left/right labels.

    python -m vision.visualize_landmarks

Landmarks are computed on the ORIGINAL frame. Only the picture is mirrored
for display (x -> 1 - x on the points). Hold up ONE hand: the label drawn
('L'/'R') shows which slot (0-20 or 21-41) it was written to.
Keys: q = quit.
"""

import time

import cv2
import numpy as np

from vision.camera import Camera
from vision.extract_landmarks import LandmarkExtractor
from vision.hand_tracker import HAND_CONNECTIONS
from vision.pose_tracker import POSE_CONNECTIONS
from vision.preprocessing import (
    LEFT_HAND, POSE, RIGHT_HAND, detected_mask, flip_for_display, mirror_points,
)


def _draw(view, pts, connections, color):
    h, w = view.shape[:2]
    xy = (mirror_points(pts)[:, :2] * [w, h]).astype(float)
    for a, b in connections:
        if not (np.isnan(xy[a]).any() or np.isnan(xy[b]).any()):
            cv2.line(view, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)),
                     color, 2)
    for p in xy:
        if not np.isnan(p).any():
            cv2.circle(view, (int(p[0]), int(p[1])), 3, color, -1)
    return xy


def main():
    extractor = LandmarkExtractor()
    prev = time.time()
    with Camera() as cam:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            lm = extractor.extract(frame)  # original frame
            view = flip_for_display(frame)  # display only

            xy_l = _draw(view, lm[LEFT_HAND], HAND_CONNECTIONS, (255, 0, 0))
            xy_r = _draw(view, lm[RIGHT_HAND], HAND_CONNECTIONS, (0, 0, 255))
            _draw(view, lm[POSE], POSE_CONNECTIONS, (0, 255, 0))
            for name, xy in (("L", xy_l), ("R", xy_r)):
                if not np.isnan(xy[0]).any():
                    cv2.putText(view, name, (int(xy[0][0]), int(xy[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            n = int(detected_mask(lm).sum())
            cv2.putText(view, f"FPS {fps:.0f}  points {n}/67", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("landmarks", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()