"""Record short webcam clips per sign (for the dataset).

    python -m vision.record_videos --label HELLO --count 5 --seconds 3

SPACE = start a clip, q = quit. Clips are saved NOT mirrored (same as the
live pipeline). The preview is mirrored for comfort only.
Output: data/videos/<LABEL>/<LABEL>_<n>.mp4
"""

import argparse
import time
from pathlib import Path

import cv2

from vision.camera import Camera
from vision.preprocessing import flip_for_display


def record(label, count, seconds, out_dir, camera_index):
    folder = Path(out_dir) / label
    folder.mkdir(parents=True, exist_ok=True)
    start_n = len(list(folder.glob(f"{label}_*.mp4")))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with Camera(camera_index) as cam:
        for i in range(start_n, start_n + count):
            # Wait for SPACE
            while True:
                ok, frame = cam.read()
                if not ok:
                    return
                view = flip_for_display(frame)
                cv2.putText(view, f"{label} #{i}: SPACE to record, q to quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.imshow("record", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                if key == ord(" "):
                    break

            h, w = frame.shape[:2]
            path = folder / f"{label}_{i}.mp4"
            writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
            t0 = time.time()
            while time.time() - t0 < seconds:
                ok, frame = cam.read()
                if not ok:
                    break
                writer.write(frame)  # original frame
                view = flip_for_display(frame)
                cv2.circle(view, (20, 20), 8, (0, 0, 255), -1)
                cv2.imshow("record", view)
                cv2.waitKey(1)
            writer.release()
            print(f"saved {path}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--out", default="data/videos")
    p.add_argument("--camera", type=int, default=0)
    a = p.parse_args()
    record(a.label, a.count, a.seconds, a.out, a.camera)


if __name__ == "__main__":
    main()