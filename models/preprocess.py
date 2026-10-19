import hashlib
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "Projects" / "asl-data"
SRC = DATA_DIR / "asl40_landmarks.npz"
DST = DATA_DIR / "asl40_processed.npz"
N_FRAMES = 32
MIN_HAND_FRAMES = 3
LEFT = slice(0, 21)
RIGHT = slice(21, 42)
NOSE = 42
L_SHOULDER = 42 + 11
R_SHOULDER = 42 + 12


def resample(x, n):
    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n)
    return np.stack([np.interp(t_new, t_old, x[:, j]) for j in range(x.shape[1])], axis=1)


def sequence_features(seq):
    seq = seq.astype(np.float32)
    left = seq[:, LEFT]
    right = seq[:, RIGHT]
    use_left = np.sum(~np.isnan(left[:, 0, 0])) > np.sum(~np.isnan(right[:, 0, 0]))
    hand = left if use_left else right
    keep = ~np.isnan(hand[:, 0, 0])
    if keep.sum() < MIN_HAND_FRAMES:
        return None
    hand = hand[keep].copy()
    nose_ref = np.nanmean(seq[keep, NOSE, :2], axis=0)
    width = np.nanmedian(np.abs(seq[keep, L_SHOULDER, 0] - seq[keep, R_SHOULDER, 0]))
    if np.isnan(nose_ref).any() or np.isnan(width) or width < 1e-3:
        return None
    if use_left:
        hand[:, :, 0] = 1.0 - hand[:, :, 0]
        nose_ref[0] = 1.0 - nose_ref[0]
    shape = hand - hand[:, :1, :]
    size = np.maximum(np.linalg.norm(shape[:, 9, :2], axis=1), 1e-6)
    shape = shape / size[:, None, None]
    location = (hand[:, 0, :2] - nose_ref) / width
    feats = np.concatenate([shape.reshape(len(hand), -1), location], axis=1)
    return resample(feats, N_FRAMES).astype(np.float32)


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