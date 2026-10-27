"""Moteur de traduction : transforme les predictions brutes en mots et en phrase.

Le vote sur N images est fait en amont par Stabilizer (translation/stabilizer.py) ;
le moteur ne garde que le seuil de confiance, l'allowlist et l'anti-doublon.
"""
import numbers
from collections import deque

from .lexicon import display_word
from .prediction import _SIGN_ID, Prediction

# Memoire bornee : la phrase ne peut jamais grossir sans limite.
MAX_WORDS = 50

# Liste officielle par defaut (S2) : les 8 mots du modele et THANKS (clavier).
# Les demos la remplacent par la liste exacte de leur predicteur.
DEFAULT_ALLOWLIST = frozenset(
    {"HELLO", "YES", "YOURSELF", "DAD", "WAIT", "WATER", "HOME", "SICK", "THANKS"}
)


def _validate_allowlist(allowlist) -> frozenset[str]:
    """Verifie l'allowlist et la renvoie figee (frozenset)."""
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
        """Traite une prediction. Renvoie le mot valide, ou None.

        Un signe hors de l'allowlist est ignore : jamais affiche, prononce
        ni stocke.
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
        """Remplace la liste (ex. par les mots du modele, une fois charge)."""
        self._allowlist = _validate_allowlist(value)

    def allows(self, sign: str) -> bool:
        """True si sign fait partie de l'allowlist."""
        if not isinstance(sign, str):
            raise TypeError(f"sign must be a str, got {type(sign).__name__}")
        return sign in self._allowlist

    @property
    def words(self) -> tuple[str, ...]:
        """Identifiants bruts des mots valides."""
        return tuple(self._words)

    @property
    def sentence(self) -> str:
        """Phrase a afficher (avec le lexique d'affichage)."""
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
        """Bouton Clear : efface la phrase."""
        self._words.clear()

    def reset(self) -> None:
        """Bouton Reset : efface la phrase et l'etat interne.

        N'affecte pas voice_enabled : c'est un reglage de l'utilisateur.
        Plus tard, reset() videra aussi le buffer de stabilisation.
        """
        self.clear()