"""Translation engine: turns raw predictions into words and a sentence.

Voting over N frames is done upstream by Stabilizer (translation/stabilizer.py);
the engine only applies the confidence threshold, the allowlist and de-duplication.
"""
import numbers
from collections import deque

from .lexicon import display_word
from .prediction import _SIGN_ID, Prediction

# Bounded memory: the sentence can never grow without limit.
MAX_WORDS = 50

# Default official list (S2): the model's 8 words plus THANKS (keyboard).
# The demos replace it with the exact list of their predictor.
DEFAULT_ALLOWLIST = frozenset(
    {"HELLO", "YES", "YOURSELF", "DAD", "WAIT", "WATER", "HOME", "SICK", "THANKS"}
)


def _validate_allowlist(allowlist) -> frozenset[str]:
    """Check the allowlist and return it frozen (frozenset)."""
    if not isinstance(allowlist, (set, frozenset, list, tuple)):
        raise TypeError("allowlist must be a set, frozenset, list or tuple of str")
    for sign in allowlist:
        if not isinstance(sign, str):
            raise TypeError(f"allowlist entries must be str, got {type(sign).__name__}")
        if not _SIGN_ID.fullmatch(sign):
            raise ValueError(f"invalid sign id in allowlist: {sign!r}")
    if not allowlist:
        raise ValueError("allowlist must not be empty")
    return frozenset(allowlist)


class TranslationEngine:
    def __init__(self, threshold=0.8, allowlist=DEFAULT_ALLOWLIST):
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            raise TypeError("threshold must be a real number")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        self.threshold = float(threshold)
        self._allowlist = _validate_allowlist(allowlist)
        self._words = deque(maxlen=MAX_WORDS)
        self._voice_enabled = True

    def update(self, prediction: Prediction | None) -> str | None:
        """Process a prediction. Return the accepted word, or None.

        A sign outside the allowlist is ignored: never displayed, spoken
        or stored.
        """
        if prediction is None:
            return None
        if not isinstance(prediction, Prediction):
            raise TypeError("update() expects a Prediction or None")
        if prediction.sign not in self._allowlist:
            return None
        if prediction.confidence < self.threshold:
            return None
        if self._words and self._words[-1] == prediction.sign:
            return None
        self._words.append(prediction.sign)
        return prediction.sign

    @property
    def allowlist(self) -> frozenset[str]:
        return self._allowlist

    @allowlist.setter
    def allowlist(self, value) -> None:
        """Replace the list (e.g. with the model's words, once it is loaded)."""
        self._allowlist = _validate_allowlist(value)

    def allows(self, sign: str) -> bool:
        """True if sign is in the allowlist."""
        if not isinstance(sign, str):
            raise TypeError(f"sign must be a str, got {type(sign).__name__}")
        return sign in self._allowlist

    @property
    def words(self) -> tuple[str, ...]:
        """Raw ids of the accepted words."""
        return tuple(self._words)

    @property
    def sentence(self) -> str:
        """Sentence to display (using the display lexicon)."""
        return " ".join(display_word(word) for word in self._words)

    @property
    def voice_enabled(self) -> bool:
        return self._voice_enabled

    @voice_enabled.setter
    def voice_enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("voice_enabled must be a bool")
        self._voice_enabled = value

    def clear(self) -> None:
        """Clear button: erase the sentence."""
        self._words.clear()

    def reset(self) -> None:
        """Reset button: erase the sentence and the internal state.

        Does not affect voice_enabled: that is a user setting.
        Later, reset() will also empty the stabilization buffer.
        """
        self.clear()