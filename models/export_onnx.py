import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from train_gru import SignGRU

DATA_DIR = Path.home() / "Projects" / "asl-data"
MODELS_DIR = Path(__file__).parent


class WithProbabilities(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="gru40")
    args = parser.parse_args()

    checkpoint = torch.load(DATA_DIR / f"{args.name}.pt", weights_only=True)
    state = checkpoint["state_dict"]
    model = SignGRU(
        checkpoint["n_features"], len(checkpoint["words"]),
        state["mean"], state["std"], checkpoint["hidden"],
    )
    model.load_state_dict(state)
    wrapper = WithProbabilities(model).eval()

    onnx_path = DATA_DIR / f"{args.name}.onnx"
    dummy = torch.zeros(1, 32, checkpoint["n_features"])
    torch.onnx.export(
        wrapper, (dummy,), str(onnx_path),
        input_names=["x"], output_names=["probs"],
        dynamic_axes={"x": {0: "batch"}, "probs": {0: "batch"}},
        external_data=False,
    )

    with np.load(DATA_DIR / "asl40_processed.npz") as f:
        sample = f["X"][:300].astype(np.float32)
    with torch.no_grad():
        expected = wrapper(torch.from_numpy(sample)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"x": sample})[0]
    same = np.mean(expected.argmax(1) == got.argmax(1)) * 100
    print("max difference PyTorch vs ONNX:", float(np.abs(expected - got).max()))
    print("same answers:", round(same, 1), "%")

    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    (MODELS_DIR / f"{args.name}.sha256").write_text(digest + "\n")
    words_path = MODELS_DIR / f"{args.name}.words.json"
    words_path.write_text(json.dumps(checkpoint["words"], indent=1) + "\n")
    print("saved:", onnx_path, round(onnx_path.stat().st_size / 1e6, 2), "MB")
    print("sha256:", digest)


if __name__ == "__main__":
    main()