import hashlib
import io
import json
import os
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

N_POINTS = 67
N_FRAMES = 32
MIN_HAND_FRAMES = 3
LIVE_MIN_HAND_FRAMES = 12
WINDOW = 30
LEFT = slice(0, 21)
RIGHT = slice(21, 42)
NOSE = 42
L_SHOULDER = 42 + 11
R_SHOULDER = 42 + 12
DEFAULT_MODEL = Path.home() / "Projects" / "asl-data" / "gru40.onnx"
MODEL_PATH = Path(os.environ.get("SIGN_MODEL_PATH", DEFAULT_MODEL))

_buffer = deque(maxlen=WINDOW)
_bundle = None


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


def _load():
    global _bundle
    if _bundle is None:
        expected = MODEL_PATH.with_suffix("").name + ".sha256"
        expected = Path(__file__).with_name(expected).read_text().strip()
        data = MODEL_PATH.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"model hash mismatch for {MODEL_PATH.name}")
        if MODEL_PATH.suffix == ".onnx":
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                data, sess_options=options, providers=["CPUExecutionProvider"]
            )
            words_file = Path(__file__).with_name(MODEL_PATH.stem + ".words.json")
            _bundle = {
                "kind": "onnx",
                "session": session,
                "words": json.loads(words_file.read_text()),
            }
        else:
            loaded = joblib.load(io.BytesIO(data))
            model = loaded["model"]
            model.n_jobs = 1
            _bundle = {
                "kind": "rf",
                "model": model,
                "words": [loaded["words"][int(c)] for c in model.classes_],
            }
    return _bundle


def words():
    return list(_load()["words"])


def reset():
    _buffer.clear()


def _to_frame(features):
    frame = np.asarray(features, dtype=np.float32)
    if frame.size != N_POINTS * 3:
        raise ValueError(f"expected {N_POINTS * 3} values, got {frame.size}")
    frame = frame.reshape(N_POINTS, 3).copy()
    if np.isinf(frame).any():
        raise ValueError("infinite values are not allowed")
    frame[np.all(frame == 0, axis=1)] = np.nan
    return frame


def _probabilities(feats):
    bundle = _load()
    if bundle["kind"] == "onnx":
        return bundle["session"].run(None, {"x": feats[None]})[0][0]
    return bundle["model"].predict_proba(feats.reshape(1, -1))[0]


def _hand_frames(seq):
    left = np.sum(~np.isnan(seq[:, LEFT][:, 0, 0]))
    right = np.sum(~np.isnan(seq[:, RIGHT][:, 0, 0]))
    return int(max(left, right))


def predict(features):
    frame = _to_frame(features)
    _buffer.append(frame)
    no_left = np.isnan(frame[LEFT, 0]).all()
    no_right = np.isnan(frame[RIGHT, 0]).all()
    if no_left and no_right:
        return None
    seq = np.stack(_buffer)
    if _hand_frames(seq) < LIVE_MIN_HAND_FRAMES:
        return None
    feats = sequence_features(seq)
    if feats is None:
        return None
    probs = _probabilities(feats)
    k = int(np.argmax(probs))
    return _load()["words"][k], float(probs[k])