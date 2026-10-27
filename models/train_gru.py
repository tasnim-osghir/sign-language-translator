import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

DATA_DIR = Path.home() / "Projects" / "asl-data"
N_TEST_PEOPLE = 4
N_VAL_PEOPLE = 2
SEED = 42
HIDDEN = 128
BATCH = 256
MIN_STD = 0.05


class SignGRU(nn.Module):
    def __init__(self, n_features, n_classes, mean, std, hidden=HIDDEN):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.gru = nn.GRU(n_features, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out, _ = self.gru(x)
        return self.head(self.drop(out.mean(dim=1)))


def augment(x):
    b, t, f = x.shape
    start = torch.rand(b, 1) * 0.15
    end = 1 - torch.rand(b, 1) * 0.15
    pos = (start + (end - start) * torch.linspace(0, 1, t)) * (t - 1)
    lo = pos.floor().long().clamp(0, t - 1)
    hi = (lo + 1).clamp(max=t - 1)
    w = (pos - lo).unsqueeze(-1)
    x_lo = torch.gather(x, 1, lo.unsqueeze(-1).expand(-1, -1, f))
    x_hi = torch.gather(x, 1, hi.unsqueeze(-1).expand(-1, -1, f))
    x = x_lo * (1 - w) + x_hi * w
    return x + torch.randn_like(x) * 0.02


def predict_all(model, X):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)])
    return logits.argmax(1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="gru40")
    parser.add_argument("--words", nargs="*")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(SEED)

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
    y = np.searchsorted(classes, y)
    print("words:", len(words), "examples:", len(y))

    people = np.unique(groups)
    rng = np.random.default_rng(SEED)
    test_people = rng.choice(people, size=N_TEST_PEOPLE, replace=False)
    rest = np.setdiff1d(people, test_people)
    val_people = rng.choice(rest, size=N_VAL_PEOPLE, replace=False)
    test = np.isin(groups, test_people)
    val = np.isin(groups, val_people)
    train = ~test & ~val
    print("test people:", sorted(test_people.tolist()))
    print("validation people:", sorted(val_people.tolist()))
    print("train / val / test examples:", int(train.sum()), int(val.sum()), int(test.sum()))

    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    Xtr, ytr = Xt[train], yt[train]
    mean = Xtr.reshape(-1, X.shape[2]).mean(0)
    std = Xtr.reshape(-1, X.shape[2]).std(0).clamp(min=MIN_STD)

    model = SignGRU(X.shape[2], len(words), mean, std)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state = -1.0, None
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(augment(Xtr[idx])), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_acc = accuracy_score(y[val], predict_all(model, Xt[val]))
        minutes = (time.time() - started) / 60
        print(
            f"epoch {epoch:2d}  loss {total / len(Xtr):.3f}  "
            f"val accuracy {val_acc * 100:.1f}%  ({minutes:.1f} min)"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"best val accuracy {best_acc * 100:.1f}%")

    pred = predict_all(model, Xt[test])
    print("TEST accuracy:", round(accuracy_score(y[test], pred) * 100, 1), "%")
    print("TEST macro F1:", round(f1_score(y[test], pred, average="macro") * 100, 1), "%")

    cm = confusion_matrix(y[test], pred, labels=range(len(words)))
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

    checkpoint = {
        "state_dict": model.state_dict(),
        "words": [w.upper() for w in words],
        "n_features": int(X.shape[2]),
        "hidden": HIDDEN,
    }
    torch.save(checkpoint, DATA_DIR / f"{args.name}.pt")
    print("saved:", DATA_DIR / f"{args.name}.pt")


if __name__ == "__main__":
    main()