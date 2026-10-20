"""Stabilisation des predictions : vote majoritaire sur les N dernieres images.

Le modele predit image par image et peut hesiter. Le Stabilizer ne renvoie un
signe que s'il est majoritaire et vu au moins min_votes fois dans la fenetre.
"""
from collections import deque

from .prediction import Prediction


class Stabilizer:
    def __init__(self, window=10, min_votes=7):
        for name, value in (("window", window), ("min_votes", min_votes)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= min_votes <= window:
            raise ValueError("min_votes must be between 1 and window")

        self.window = window
        self.min_votes = min_votes
        # Une entree par image : la Prediction, ou None si rien n'a ete predit.
        self._entries = deque(maxlen=window)

    def update(self, prediction: Prediction | None) -> Prediction | None:
        """Ajoute une prediction a la fenetre. Renvoie le signe stable, ou None.

        Le signe stable est celui qui a le plus de votes (en cas d'egalite, le
        plus recent) ; sa confiance est la moyenne de ses entrees.
        """
        if prediction is not None and not isinstance(prediction, Prediction):
            raise TypeError("update() expects a Prediction or None")
        self._entries.append(prediction)

        confidences = {}
        last_seen = {}
        for index, entry in enumerate(self._entries):
            if entry is None:
                continue
            confidences.setdefault(entry.sign, []).append(entry.confidence)
            last_seen[entry.sign] = index
        if not confidences:
            return None

        sign = max(confidences, key=lambda s: (len(confidences[s]), last_seen[s]))
        votes = confidences[sign]
        if len(votes) < self.min_votes:
            return None
        return Prediction(sign, sum(votes) / len(votes))

    def reset(self) -> None:
        """Vide la fenetre."""
        self._entries.clear()
