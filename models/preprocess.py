import hashlib
from pathlib import Path

import numpy as np
from predictor import sequence_features

DATA_DIR = Path.home() / "Projects" / "asl-data"
SRC = DATA_DIR / "asl40_landmarks.npz"
DST = DATA_DIR / "asl40_processed.npz"


def main():
    with np.load(SRC) as f:
        data = f["data"]
        offsets = f["offsets"]
        labels = f["labels"]
        participants = f["participants"]
    X, y, groups = [], [], []
    dropped = 0
    for i in range(len(labels)):
        feats = sequence_features(data[offsets[i]:offsets[i + 1]])
        if feats is None:
            dropped += 1
            continue
        X.append(feats)
        y.append(labels[i])
        groups.append(participants[i])
    X = np.stack(X)
    y = np.array(y)
    groups = np.array(groups)
    print("kept:", len(y), "dropped:", dropped)
    print("X shape:", X.shape)
    print("NaN left:", bool(np.isnan(X).any()))
    np.savez_compressed(DST, X=X, y=y, groups=groups)
    print("saved:", round(DST.stat().st_size / 1e6, 1), "MB")
    print("sha256:", hashlib.sha256(DST.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()