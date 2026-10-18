"""Moteur de traduction : transforme les predictions brutes en mots et en phrase.

Version squelette (semaine 1) : l'interface est definitive, la logique interne
est volontairement simple. La stabilisation (vote sur N images, delai
anti-repetition) arrive en semaine 2 ; window, min_votes et cooldown_s sont
deja valides et stockes, mais pas encore utilises.
"""
import numbers
from collections import deque

from .prediction import Prediction

# Memoire bornee : la phrase ne peut jamais grossir sans limite.
MAX_WORDS = 50


class TranslationEngine:
    def __init__(self, threshold=0.8, window=10, min_votes=7, cooldown_s=1.5):
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            raise TypeError("threshold must be a real number")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        for name, value in (("window", window), ("min_votes", min_votes)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= min_votes <= window:
            raise ValueError("min_votes must be between 1 and window")
        if isinstance(cooldown_s, bool) or not isinstance(cooldown_s, numbers.Real):
            raise TypeError("cooldown_s must be a real number")
        if not cooldown_s >= 0:
            raise ValueError("cooldown_s must be >= 0")

        self.threshold = float(threshold)
        self.window = window
        self.min_votes = min_votes
        self.cooldown_s = float(cooldown_s)
        self._words = deque(maxlen=MAX_WORDS)
        self._voice_enabled = True

    def update(self, prediction: Prediction | None) -> str | None:
        """Traite une prediction. Renvoie le mot valide, ou None."""
        if prediction is None:
            return None
        if not isinstance(prediction, Prediction):
            raise TypeError("update() expects a Prediction or None")
        if prediction.confidence < self.threshold:
            return None
        if self._words and self._words[-1] == prediction.sign:
            return None
        self._words.append(prediction.sign)
        return prediction.sign

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(self._words)

    @property
    def sentence(self) -> str:
        return " ".join(self._words)

    @property
    def voice_enabled(self) -> bool:
        return self._voice_enabled

    @voice_enabled.setter
    def voice_enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("voice_enabled must be a bool")
        self._voice_enabled = value

    def clear(self) -> None:
        """Bouton Clear : efface la phrase."""
        self._words.clear()

    def reset(self) -> None:
        """Bouton Reset : efface la phrase et l'etat interne.

        N'affecte pas voice_enabled : c'est un reglage de l'utilisateur.
        Plus tard, reset() videra aussi le buffer de stabilisation.
        """
        self.clear()