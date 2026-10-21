
"""Capture webcam frames and save combined sign-language landmarks."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from vision.hand_tracker import HandTracker
from vision.pose_tracker import PoseTracker
from vision.preprocessing import combine_landmarks


def extract_from_camera(camera_index=0, output_path="landmarks.npy",
                        show_video=True):
    """
    Capture webcam frames and save:
        landmarks.npy:       (number_of_frames, 67, 3)
        landmarks_flat.npy:  (number_of_frames, 201)

    Press Q to stop capture.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}."
        )

    hand_tracker = HandTracker()
    pose_tracker = PoseTracker()

    sequence = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read a frame from the camera.")
                break

            # IMPORTANT: detect on the original, unflipped frame.
            hand_data = hand_tracker.process(frame)
            pose_data = pose_tracker.process(frame)

            landmarks = combine_landmarks(
                left_hand=hand_data["left_hand"],
                right_hand=hand_data["right_hand"],
                pose=pose_data["pose"],
            )

            sequence.append(landmarks)

            if show_video:
                display = frame.copy()
                hand_tracker.draw(display, hand_data["results"])
                pose_tracker.draw(display, pose_data["results"])

                # Mirror the display only; do not feed it to the trackers.
                display = cv2.flip(display, 1)

                cv2.putText(
                    display,
                    f"Shape: {landmarks.shape} | Q: stop",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Landmark extraction", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        hand_tracker.close()
        pose_tracker.close()
        cv2.destroyAllWindows()

    if not sequence:
        print("No frames captured. Nothing was saved.")
        return None

    landmarks_sequence = np.stack(sequence, axis=0)
    flattened_sequence = landmarks_sequence.reshape(len(sequence), 201)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    flat_output = output.with_name(
        f"{output.stem}_flat{output.suffix}"
    )

    np.save(output, landmarks_sequence)
    np.save(flat_output, flattened_sequence)

    print(f"Frames captured: {len(sequence)}")
    print(f"Landmarks saved: {output}")
    print(f"Sequence shape: {landmarks_sequence.shape}")
    print(f"Flat values saved: {flat_output}")
    print(f"Flat shape: {flattened_sequence.shape}")

    return landmarks_sequence


def main():
    parser = argparse.ArgumentParser(
        description="Extract sign-language landmarks from a webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/landmarks.npy",
        help="Output .npy file path",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Capture without showing the video window",
    )

    args = parser.parse_args()

    extract_from_camera(
        camera_index=args.camera,
        output_path=args.output,
        show_video=not args.no_display,
    )


if __name__ == "__main__":
    main()
