"""Display lexicon: turns a sign id into readable text.

Display only: it is applied after the allowlist, on words that are already
accepted. A word missing from the lexicon is shown as is.
"""

DISPLAY_LEXICON = {
    "THANKYOU": "thank you",
    "MINEMY": "my",
    "HESHEIT": "he/she/it",
    "WEUS": "we",
}


def display_word(sign: str) -> str:
    """Return the text to display for a sign id."""
    if not isinstance(sign, str):
        raise TypeError(f"sign must be a str, got {type(sign).__name__}")
    return DISPLAY_LEXICON.get(sign, sign)
