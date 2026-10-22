"""Record short webcam clips per sign (for the dataset).

Usage:
    python -m vision.record_videos --label HELLO --count 5 --seconds 3

SPACE = start a clip, q = quit. Clips are saved NOT mirrored.
The preview is mirrored for comfort only.
Output: data/videos/<LABEL>/<LABEL>_<n>.mp4
"""

import argparse
import time
from pathlib import Path

import cv2

from vision.camera import Camera


def record(label, count, seconds, out_dir, camera_index):
    folder = Path(out_dir) / label
    folder.mkdir(parents=True, exist_ok=True)

    start_n = len(list(folder.glob(f"{label}_*.mp4")))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with Camera(camera_index) as cam:
        for i in range(start_n, start_n + count):
            # Wait for SPACE to start recording.
            while True:
                success, frame = cam.read()

                if not success or frame is None:
                    continue

                view = cv2.flip(frame, 1)
                cv2.putText(
                    view,
                    f"{label} #{i}: SPACE to record, q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("record", view)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                if key == ord(" "):
                    break

            height, width = frame.shape[:2]
            path = folder / f"{label}_{i}.mp4"
            writer = cv2.VideoWriter(
                str(path), fourcc, 30.0, (width, height)
            )

            if not writer.isOpened():
                print(f"Error: could not create video file {path}")
                continue

            try:
                start_time = time.time()

                while time.time() - start_time < seconds:
                    success, frame = cam.read()

                    if not success or frame is None:
                        continue

                    writer.write(frame)

                    view = cv2.flip(frame, 1)
                    cv2.circle(view, (20, 20), 8, (0, 0, 255), -1)
                    cv2.imshow("record", view)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
            finally:
                writer.release()

            print(f"Saved {path}")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out", default="data/videos")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    record(
        args.label,
        args.count,
        args.seconds,
        args.out,
        args.camera,
    )


if __name__ == "__main__":
    main()
