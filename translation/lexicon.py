"""Lexique d'affichage : transforme un identifiant de signe en texte lisible.

Affichage seulement : il s'applique apres l'allowlist, sur des mots deja
valides. Un mot absent du lexique est affiche tel quel.
"""

DISPLAY_LEXICON = {
    "THANKYOU": "thank you",
    "MINEMY": "my",
    "HESHEIT": "he/she/it",
    "WEUS": "we",
}


def display_word(sign: str) -> str:
    """Renvoie le texte a afficher pour un identifiant de signe."""
    if not isinstance(sign, str):
        raise TypeError(f"sign must be a str, got {type(sign).__name__}")
    return DISPLAY_LEXICON.get(sign, sign)
