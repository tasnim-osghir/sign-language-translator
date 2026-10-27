"""S4: a model file whose hash does not match is refused (Member 5)."""
import pytest

from models import predictor


def test_tampered_model_is_refused(tmp_path, monkeypatch):
    fake = tmp_path / "rf_demo.joblib"
    fake.write_bytes(b"this is not the real model")
    monkeypatch.setattr(predictor, "MODEL_PATH", fake)
    monkeypatch.setattr(predictor, "_bundle", None)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        predictor._load()