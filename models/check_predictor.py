import json
import time
from pathlib import Path

import numpy as np
import predictor

DATA_DIR = Path.home() / "Projects" / "asl-data"
DEMO_WORDS = ["hello", "yes", "yourself", "water", "home", "sick", "dad", "wait"]
PER_WORD = 5


def main():
    with np.load(DATA_DIR / "asl40_landmarks.npz") as f:
        data = f["data"]
        offsets = f["offsets"]
        labels = f["labels"]
    with open(DATA_DIR / "asl40_labels.json") as fh:
        label_map = json.load(fh)

    print("model words:", predictor.words())
    correct = 0
    total = 0
    times = []
    for word in DEMO_WORDS:
        for i in np.flatnonzero(labels == label_map[word])[:PER_WORD]:
            predictor.reset()
            last = None
            for frame in data[offsets[i]:offsets[i + 1]]:
                start = time.perf_counter()
                result = predictor.predict(frame.reshape(-1))
                times.append(time.perf_counter() - start)
                if result is not None:
                    last = result
            total += 1
            correct += last is not None and last[0] == word.upper()
            print(f"{word:<10} -> {last}")
    print(f"correct: {correct}/{total}")
    print(f"time per frame: {np.mean(times) * 1000:.1f} ms, max {np.max(times) * 1000:.1f} ms")

    try:
        predictor.predict([0.5] * 10)
    except ValueError as error:
        print("bad input rejected:", error)


if __name__ == "__main__":
    main()