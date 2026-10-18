import json
from pathlib import Path
import numpy as np
import mediapipe
import cv2
import sklearn

DATA_DIR = Path.home() / "Projects" / "asl-data"

print("mediapipe", mediapipe.__version__)
print("opencv", cv2.__version__)
print("scikit-learn", sklearn.__version__)

with np.load(DATA_DIR / "asl40_landmarks.npz") as f:
    data = f["data"]
    offsets = f["offsets"]
    labels = f["labels"]
    participants = f["participants"]

with open(DATA_DIR / "asl40_labels.json") as fh:
    label_map = json.load(fh)

words = {v: k for k, v in label_map.items()}
lengths = np.diff(offsets)

print("all frames:", data.shape)
print("examples:", len(labels), "words:", len(label_map), "people:", len(np.unique(participants)))
print("frames per example: min", lengths.min(), "median", int(np.median(lengths)), "max", lengths.max())

first = data[offsets[0]:offsets[1]]
print("first example:", words[int(labels[0])], first.shape)

left = ~np.isnan(data[:, 0, 0])
right = ~np.isnan(data[:, 21, 0])
print("frames with left hand:", round(left.mean() * 100, 1), "%")
print("frames with right hand:", round(right.mean() * 100, 1), "%")
print("frames with no hand:", round((~left & ~right).mean() * 100, 1), "%")