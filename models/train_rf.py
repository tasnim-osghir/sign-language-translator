import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATA_DIR = Path.home() / "Projects" / "asl-data"
N_TEST_PEOPLE = 4
SEED = 42


def main():
    with np.load(DATA_DIR / "asl40_processed.npz") as f:
        X = f["X"]
        y = f["y"]
        groups = f["groups"]
    with open(DATA_DIR / "asl40_labels.json") as fh:
        label_map = json.load(fh)
    words = [w for w, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    people = np.unique(groups)
    rng = np.random.default_rng(SEED)
    test_people = rng.choice(people, size=N_TEST_PEOPLE, replace=False)
    test = np.isin(groups, test_people)
    print("test people:", sorted(test_people.tolist()))
    print("train examples:", int((~test).sum()), "test examples:", int(test.sum()))

    X_flat = X.reshape(len(X), -1)
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    model.fit(X_flat[~test], y[~test])
    pred = model.predict(X_flat[test])

    print("accuracy:", round(accuracy_score(y[test], pred) * 100, 1), "%")
    print("macro F1:", round(f1_score(y[test], pred, average="macro") * 100, 1), "%")

    cm = confusion_matrix(y[test], pred, labels=range(len(words)))
    per_word = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print("hardest words:")
    for k in np.argsort(per_word)[:8]:
        off = cm[k].copy()
        off[k] = 0
        print(f"  {words[k]:<10} {per_word[k] * 100:5.1f}%  most confused with: {words[int(off.argmax())]}")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), cmap="Blues")
    ax.set_xticks(range(len(words)), words, rotation=90)
    ax.set_yticks(range(len(words)), words)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    out = DATA_DIR / "confusion_rf.png"
    fig.savefig(out, dpi=120)
    print("confusion matrix saved:", out)


if __name__ == "__main__":
    main()