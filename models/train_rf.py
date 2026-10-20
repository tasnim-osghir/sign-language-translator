import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

DATA_DIR = Path.home() / "Projects" / "asl-data"
MODELS_DIR = Path(__file__).parent
N_TEST_PEOPLE = 4
SEED = 42


def make_model():
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="rf40")
    parser.add_argument("--words", nargs="*")
    args = parser.parse_args()

    with np.load(DATA_DIR / "asl40_processed.npz") as f:
        X = f["X"]
        y = f["y"]
        groups = f["groups"]
    with open(DATA_DIR / "asl40_labels.json") as fh:
        label_map = json.load(fh)
    all_words = [w for w, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    if args.words:
        unknown = [w for w in args.words if w not in label_map]
        if unknown:
            raise SystemExit(f"unknown words: {unknown}")
        keep = np.isin(y, [label_map[w] for w in args.words])
        X, y, groups = X[keep], y[keep], groups[keep]
    classes = np.unique(y)
    words = [all_words[c] for c in classes]
    print("words:", len(words), "examples:", len(y))

    people = np.unique(groups)
    rng = np.random.default_rng(SEED)
    test_people = rng.choice(people, size=N_TEST_PEOPLE, replace=False)
    test = np.isin(groups, test_people)
    print("test people:", sorted(test_people.tolist()))

    X_flat = X.reshape(len(X), -1)
    model = make_model()
    model.fit(X_flat[~test], y[~test])
    pred = model.predict(X_flat[test])
    print("accuracy:", round(accuracy_score(y[test], pred) * 100, 1), "%")
    print("macro F1:", round(f1_score(y[test], pred, average="macro") * 100, 1), "%")

    cm = confusion_matrix(y[test], pred, labels=classes)
    per_word = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print("accuracy per word (best first):")
    for k in np.argsort(-per_word):
        off = cm[k].copy()
        off[k] = 0
        confused = words[int(off.argmax())] if off.sum() else "-"
        print(f"  {words[k]:<10} {per_word[k] * 100:5.1f}%  confused with: {confused}")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), cmap="Blues")
    ax.set_xticks(range(len(words)), words, rotation=90)
    ax.set_yticks(range(len(words)), words)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(DATA_DIR / f"confusion_{args.name}.png", dpi=120)

    final = make_model()
    final.fit(X_flat, y)
    model_path = DATA_DIR / f"{args.name}.joblib"
    bundle = {"model": final, "words": [w.upper() for w in all_words]}
    joblib.dump(bundle, model_path, compress=3)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (MODELS_DIR / f"{args.name}.sha256").write_text(digest + "\n")
    print("model saved:", model_path, round(model_path.stat().st_size / 1e6, 1), "MB")
    print("sha256:", digest)


if __name__ == "__main__":
    main()