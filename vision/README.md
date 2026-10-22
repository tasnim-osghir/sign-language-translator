# vision/ — Computer Vision Module

This module captures webcam frames, extracts hand and pose landmarks using MediaPipe Holistic, and prepares the output for `models/predictor.py`.

**Pipeline:** Webcam → MediaPipe Holistic → `(67, 3)` landmarks

## Output Format

For each frame, `LandmarkExtractor.extract(frame_bgr)` in `landmarks.py` returns a NumPy array with shape `(67, 3)` and data type `float32`.

The array contains raw MediaPipe landmark coordinates. No normalization is applied.

| Rows | Description |
|---|---|
| 0–20 | Left hand landmarks (21 points) |
| 21–41 | Right hand landmarks (21 points) |
| 42–66 | First 25 pose landmarks |

For the pose landmarks, row 42 corresponds to the nose, and rows 53 and 54 correspond to the shoulders.

If a landmark or body part is not detected, its coordinates are represented by `NaN`.

## Important Rules

- Pass the original camera frame to the landmark extractor.
- Use `cv2.flip()` for display purposes only. Do not flip the input frame before extraction unless the project requires it.
- Left/right hand assignment is controlled by `SWAP_HANDS` in `landmarks.py`.
- To verify hand assignment, run `python -m vision.visualize_landmarks`. Raise your right hand and check that the displayed label is `R`.
- The module requires `mediapipe==1.1.1` and the Holistic model file.
- The model path can be configured using the `SIGN_HOLISTIC_MODEL` environment variable. The default path is `~/Projects/asl-data/holistic_landmarker.task`.

## Requirements

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Make sure the Holistic model file exists at the configured path before running the landmark extractor or visualization.

## How to Run

Run the commands below from the repository root using Python's `-m` option.

### 1. Visualize landmarks

```bash
python -m vision.visualize_landmarks
```

This command opens the webcam and displays detected landmarks with left/right hand labels.

Press `q` to quit.

### 2. Record videos for a sign

```bash
python -m vision.record_videos --label HELLO --count 5
```

This command records video clips for the specified sign label. Press Space to start each clip and `q` to quit.

## Files

| File | Role |
|---|---|
| `camera.py` | Webcam wrapper. `read()` returns `(success, frame)`. |
| `landmarks.py` | `LandmarkExtractor`: extracts a `(67, 3)` landmark array from a frame. |
| `visualize_landmarks.py` | Displays live landmarks and left/right hand labels. |
| `record_videos.py` | Records video clips for individual sign labels. |

## Troubleshooting

- **Model file not found:** Check that the model file exists and that `SIGN_HOLISTIC_MODEL` points to the correct path.
- **Camera cannot open:** Check that the webcam is available and not being used by another application.
- **No hand landmarks detected:** Make sure your hands are visible and lighting is sufficient.
- **Left/right labels appear reversed:** Check `SWAP_HANDS` in `landmarks.py` and test again.
- **Missing dependencies:** Activate the project virtual environment and install dependencies from `requirements.txt`.
