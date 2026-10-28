"""Shared format of a model prediction (contract between modules)."""
import math
import numbers
import re
from dataclasses import dataclass

# Sign id: ASCII uppercase letters, digits and "_", 32 characters max.
_SIGN_ID = re.compile(r"[A-Z][A-Z0-9_]{0,31}")


@dataclass(frozen=True)
class Prediction:
    sign: str          # ASCII id, e.g. "HELLO"
    confidence: float  # between 0.0 and 1.0

    def __post_init__(self):
        # Wrong type -> TypeError; right type but forbidden value -> ValueError.
        if not isinstance(self.sign, str):
            raise TypeError(f"sign must be a str, got {type(self.sign).__name__}")
        if not _SIGN_ID.fullmatch(self.sign):
            raise ValueError(f"invalid sign id: {self.sign!r}")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, numbers.Real
        ):
            kind = type(self.confidence).__name__
            raise TypeError(f"confidence must be a real number, got {kind}")
        if math.isnan(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range [0, 1]: {self.confidence!r}")
        # Force a real float (the model may return a NumPy float32).
        object.__setattr__(self, "confidence", float(self.confidence))