from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "Projects" / "asl-data"
LEFT_WRIST = 0
RIGHT_WRIST = 21
L_SHOULDER = 53
R_SHOULDER = 54


def main():
    with np.load(DATA_DIR / "asl40_landmarks.npz") as f:
        data = f["data"].astype(np.float32)
    ls = data[:, L_SHOULDER, 0]
    rs = data[:, R_SHOULDER, 0]
    body = ~np.isnan(ls) & ~np.isnan(rs)
    print("frames with both shoulders:", int(body.sum()))
    mirrored = np.mean(rs[body] > ls[body]) * 100
    print("right shoulder on the image's RIGHT side:", round(mirrored, 1), "%")
    for name, wrist in (("rows 0-20", LEFT_WRIST), ("rows 21-41", RIGHT_WRIST)):
        x = data[:, wrist, 0]
        ok = body & ~np.isnan(x)
        near_left = np.mean(np.abs(x[ok] - ls[ok]) < np.abs(x[ok] - rs[ok])) * 100
        print(f"{name}: {int(ok.sum())} frames, closer to the LEFT shoulder: {round(near_left, 1)} %")


if __name__ == "__main__":
    main()